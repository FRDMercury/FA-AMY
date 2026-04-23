import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ======================================================
# CONFIGURATION AREA - PLEASE MODIFY THE PATHS BELOW
# ======================================================
# Path to your pre-computed embedding features (.npy file)
INPUT_NPY_PATH = "Test.npy"

# Path to save the final prediction results
OUTPUT_CSV_PATH = "prediction_results.csv"

# Path to your trained model checkpoint (.pth file)
MODEL_CHECKPOINT_PATH = "final_model.pth"

# Inference Settings
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================

def set_seed(seed=777):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(777)


class EmbeddingDataset(Dataset):
    """Dataset class for loading pre-computed protein embeddings."""

    def __init__(self, data_path):
        # Expecting data shape: [N, Sequence_Length, Embedding_Dim]
        self.data = np.load(data_path)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float)

    def __len__(self):
        return len(self.data)


# ================== Model Architecture ==================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        return x + res


class TCNBlock(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                ResidualBlock(input_dim if i == 0 else num_channels[i - 1],
                              out_channels, kernel_size, dilation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BiTCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5, merge_mode='concat'):
        super().__init__()
        self.forward_tcn = TCNBlock(input_dim, num_channels, kernel_size)
        self.backward_tcn = TCNBlock(input_dim, num_channels, kernel_size)
        self.merge_mode = merge_mode

    def forward(self, x):
        x_rev = torch.flip(x, dims=[-1])
        out_fwd = self.forward_tcn(x)
        out_bwd = self.backward_tcn(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[-1])
        if self.merge_mode == 'concat':
            out = torch.cat([out_fwd, out_bwd], dim=1)
        elif self.merge_mode == 'sum':
            out = out_fwd + out_bwd
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, D = x.size()
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        out = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class LIA1D(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, f, kernel_size=1)
        self.softpool = nn.AvgPool1d(kernel_size=7, stride=3, padding=0)
        self.conv2 = nn.Conv1d(f, f, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(f, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(nn.Sigmoid())
        kernel = torch.ones(3) / 3.0
        self.register_buffer('smooth_kernel', kernel.view(1, 1, 3))

    def forward(self, x):
        x = x.transpose(1, 2)
        g = self.gate(x[:, :1])
        w = self.conv1(x);
        w = self.softpool(w);
        w = self.conv2(w);
        w = self.conv3(w)
        w = self.sigmoid(w)
        w = F.interpolate(w, size=x.size(2), mode='nearest')
        w = F.conv1d(w, self.smooth_kernel.repeat(w.size(1), 1, 1),
                     padding=1, groups=w.size(1))
        out = x * w * g
        return out.transpose(1, 2)


class FAAmyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Input dim 1152 corresponds to ESMC-600m embeddings
        self.bitcn = BiTCN(input_dim=1152, num_channels=[512, 256, 128])
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.3)
        self.att1 = MultiHeadSelfAttention(input_dim=256, num_heads=4)
        self.att2 = LIA1D(256, f=64)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # x: [Batch, Length, Dim]
        x = self.bitcn(x.permute(0, 2, 1))  # TCN expects [B, C, L]
        x = x.permute(0, 2, 1)  # Back to [B, L, C]

        attn1 = self.att1(x)
        attn2 = self.att2(x)
        fused = 0.5 * attn1 + 0.5 * attn2

        pooled = torch.mean(x + fused, dim=1)

        x = self.relu(self.fc2(pooled))
        x = self.drop(x)
        x = self.relu(self.fc3(x))
        x = self.drop(x)
        return self.fc4(x)


# ================== Inference Logic ==================

def run_prediction():
    print(f"--- FA-Amy Inference Tool ---")
    print(f"Loading features from: {INPUT_NPY_PATH}")

    # Check if files exist
    if not os.path.exists(INPUT_NPY_PATH):
        print(f"Error: {INPUT_NPY_PATH} not found.")
        return

    # Initialize Dataset and DataLoader
    dataset = EmbeddingDataset(INPUT_NPY_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = FAAmyModule().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Successfully loaded model weights from: {MODEL_CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()
    all_probs = []

    print(f"Starting inference on {DEVICE}...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(DEVICE)
            outputs = model(batch_data)
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)

            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    # Save results to CSV
    print(f"Writing results to: {OUTPUT_CSV_PATH}")
    with open(OUTPUT_CSV_PATH, "w") as f:
        f.write("Index,Probability,Prediction\n")
        for i, prob in enumerate(all_probs):
            pred = 1 if prob >= 0.5 else 0
            f.write(f"{i},{prob:.4f},{pred}\n")

    print(f"Done! Processed {len(all_probs)} sequences.")


if __name__ == "__main__":
    run_prediction()