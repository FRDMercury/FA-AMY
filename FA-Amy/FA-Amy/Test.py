import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score

# ======================================================
# CONFIGURATION AREA - PLEASE MODIFY THE PATHS BELOW
# ======================================================
# 1. Path to pre-computed positive and negative embeddings (.npy)
POS_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Generalized_dataset", "esmc_pos_test.npy")
NEG_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Generalized_dataset", "esmc_neg_test.npy")

# 2. Path to the saved model weight (.pth)
MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "Model-saved", "final_model.pth")

# 3. Hyperparameters
BATCH_SIZE = 8
SEED = 777
# ======================================================

def set_random_seed(seed):
    """Ensure reproducibility by fixing random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set deterministic algorithms (required for reproducibility in some PyTorch versions)
    torch.use_deterministic_algorithms(True, warn_only=True)

set_random_seed(SEED)

class BioinformaticsDataset(Dataset):
    """Custom dataset for loading protein embeddings and labels."""
    def __init__(self, label, prot):
        self.lb = label
        self.df_prot = prot

    def __getitem__(self, index):
        prot = torch.tensor(self.df_prot[index], dtype=torch.float)
        label = torch.tensor(self.lb[index], dtype=torch.float)
        return prot, label

    def __len__(self):
        return len(self.df_prot)

# ================== Model Architecture ==================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size - 1) * dilation // 2, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        x = self.norm(self.relu(self.conv1(x)))
        return x + res

class TCNBlock(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(ResidualBlock(input_dim if i == 0 else num_channels[i - 1], out_channels, kernel_size, dilation))
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
        out_bwd = torch.flip(self.backward_tcn(x_rev), dims=[-1])
        return torch.cat([out_fwd, out_bwd], dim=1) if self.merge_mode == 'concat' else out_fwd + out_bwd

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
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class LIA1D(nn.Module):
    def __init__(self, channels, f=64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, f, kernel_size=1)
        self.softpool = nn.AvgPool1d(kernel_size=7, stride=3, padding=0)
        self.conv2 = nn.Conv1d(f, f, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(f, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(nn.Sigmoid())
        self.register_buffer('smooth_kernel', (torch.ones(3) / 3.0).view(1, 1, 3))

    def forward(self, x):
        x = x.transpose(1, 2)
        g = self.gate(x[:, :1])
        w = self.sigmoid(self.conv3(self.conv2(self.softpool(self.conv1(x)))))
        w = F.interpolate(w, size=x.size(2), mode='nearest')
        w = F.conv1d(w, self.smooth_kernel.repeat(w.size(1), 1, 1), padding=1, groups=w.size(1))
        return (x * w * g).transpose(1, 2)

class FAAmyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Input dim 1152 for ESMC embeddings
        self.bitcn = BiTCN(input_dim=1152, num_channels=[512, 256, 64])
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.3)
        self.att1 = MultiHeadSelfAttention(input_dim=128, num_heads=4)
        self.att2 = LIA1D(128, f=64)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.bitcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        fused = 0.5 * self.att1(x) + 0.5 * self.att2(x)
        pooled = torch.mean(x + fused, dim=1)
        x = self.drop(self.relu(self.fc2(pooled)))
        x = self.drop(self.relu(self.fc3(x)))
        return self.fc4(x)

# ================== Evaluation Loop ==================

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Test on Device: {device} ---")

    # 1. Load Pre-computed Data
    try:
        pos_test = np.load(POS_DATA_PATH)
        neg_test = np.load(NEG_DATA_PATH)
        print(f"Data Loaded: Positive={len(pos_test)}, Negative={len(neg_test)}")
    except FileNotFoundError as e:
        print(f"Error: Could not find .npy files. {e}")
        return

    # Combine data and create labels
    test_prot = np.concatenate((pos_test, neg_test), axis=0)
    test_labels = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))))

    # 2. Data Loader
    test_loader = DataLoader(BioinformaticsDataset(test_labels, test_prot), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    model = FAAmyModule().to(device)
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
        print(f"Model loaded successfully from: {MODEL_CHECKPOINT_PATH}")
    else:
        print(f"Warning: Checkpoint not found at {MODEL_CHECKPOINT_PATH}. Using random weights.")

    model.eval()
    y_true, y_probs = [], []

    print("Evaluating...")
    with torch.no_grad():
        for prot_x, labels_y in test_loader:
            prot_x = prot_x.to(device)
            outputs = torch.sigmoid(model(prot_x))
            y_true.extend(labels_y.numpy())
            y_probs.extend(outputs.cpu().numpy())

    # 4. Metric Calculation
    labels = np.array(y_true)
    probs = np.array(y_probs).flatten()
    preds = np.around(probs)

    auc_score = roc_auc_score(labels, probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Calculate Metrics
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fn + fp)
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    f1 = 2 * pre * sn / (pre + sn) if (pre + sn) > 0 else 0

    print("\n" + "="*40)
    print(f"TEST RESULTS:")
    print(f"SN: {sn:.4f} | SP: {sp:.4f} | ACC: {acc:.4f}")
    print(f"MCC: {mcc:.4f} | Pre: {pre:.4f} | AUC: {auc_score:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    run_test()
