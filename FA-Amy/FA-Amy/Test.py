from torch.utils.data import Dataset, DataLoader
import os
import torch
torch.use_deterministic_algorithms(True, warn_only=False)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
import random
import numpy as np



def random_seed(seed):  # Fixed random seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


random_seed(777)


class BioinformaticsDataset(Dataset):  #
    def __init__(self, label, prot):
        self.lb = label
        self.df_prot = prot

    def __getitem__(self, index):
        prot = self.df_prot[index]
        prot = torch.tensor(prot, dtype=torch.float)
        label = self.lb[index]
        label = torch.tensor(label, dtype=torch.float)
        data_length = 902  # The features extracted by ESM C have padding before and after
        return prot, label, data_length

    def __len__(self):
        return len(self.df_prot)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation // 2,
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
                ResidualBlock(input_dim if i == 0 else num_channels[i - 1], out_channels, kernel_size, dilation))
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
        # x: (batch_size, channels, seq_len)
        x_rev = torch.flip(x, dims=[-1])
        out_fwd = self.forward_tcn(x)
        out_bwd = self.backward_tcn(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[-1])
        if self.merge_mode == 'concat':
            out = torch.cat([out_fwd, out_bwd], dim=1)
        elif self.merge_mode == 'sum':
            out = out_fwd + out_bwd
        else:
            raise ValueError("merge_mode must be 'concat' or 'sum'")
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.size()

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, D/H]

        # concat
        out = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_proj(out)  #  [B, L, D]
        return out


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
        self.register_buffer('smooth_kernel', kernel.view(1, 1, 3))  # shape: [1, 1, 3]

    def forward(self, x):
        # x: [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)  # [B, D, L]
        g = self.gate(x[:, :1])  # Gate Control

        w = self.conv1(x)  # [B, f, L]
        w = self.softpool(w)  # ↓
        w = self.conv2(w)  # ↓
        w = self.conv3(w)  # [B, D, l']
        w = self.sigmoid(w)

        # Nearest Neighbor Interpolation (Reproducible)+Smooth Convolution
        w = F.interpolate(w, size=x.size(2), mode='nearest')
        w = F.conv1d(w, self.smooth_kernel.repeat(w.size(1), 1, 1), padding=1, groups=w.size(1))

        out = x * w * g
        return out.transpose(1, 2)  # → [B, L, D]


class FAAmyModule(nn.Module):
    def __init__(self):
        super(FAAmyModule, self).__init__()
        self.bitcn = BiTCN(input_dim=1152, num_channels=[512, 256, 64], kernel_size=5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.3)
        self.att1 = MultiHeadSelfAttention(input_dim=128, num_heads=4)
        self.att2 = LIA1D(128, f=64)
        self.relu = nn.LeakyReLU()

    def forward(self, prot0):
        prot2 = self.bitcn(prot0.permute(0, 2, 1))  # [B, 128, L]
        prot2 = prot2.permute(0, 2, 1)
        atten1 = self.att1(prot2)
        atten2 = self.att2(prot2)
        fused = 0.5 * atten1 + 0.5 * atten2
        pooled = prot2 + fused
        pooled = torch.mean(pooled, dim=1)
        x = self.fc2(pooled)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x


def run_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "esmc_neg_test.npy")
    pos_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "esmc_pos_test.npy")
    model_path = os.path.join(base_dir, "..", "Model-saved")
    neg_test = np.load(neg_data_path)
    pos_test = np.load(pos_data_path)
    df_prot = np.concatenate((pos_test, neg_test), axis=0)
    ones = np.ones(len(pos_test), dtype=int)
    zeros = np.zeros(len(neg_test), dtype=int)
    df_lb = np.concatenate((ones, zeros))

    # Dataset and DataLoader Initialization
    test_set = BioinformaticsDataset(df_lb, df_prot)
    test_load = DataLoader(dataset=test_set, batch_size=8, shuffle=False)

    # Model Initialization
    model = FAAmyModule()
    model = model.to(device)

    print("==========================Test RESULT================================")
    final_model_path = os.path.join(model_path, f"final_best_model.pth")
    model.load_state_dict(
        torch.load(final_model_path, map_location=device))
    model.eval()

    arr_labels = []
    arr_labels_hyps = []



    with torch.no_grad():
        for prot_x, data_y, length in test_load:
            prot_x, data_y, length = prot_x.to(device), data_y.to(device), length.to(device)
            y_pred = torch.sigmoid(model(prot_x))
            arr_labels.extend(data_y.cpu().numpy())
            arr_labels_hyps.extend(y_pred.cpu().numpy())

    labels = np.array(arr_labels)
    probs = np.array(arr_labels_hyps)
    AUC = roc_auc_score(labels, probs)
    pred = np.around(probs)
    confus_matrix = confusion_matrix(labels, pred)
    TN, FP, FN, TP = confus_matrix.ravel()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    BA = (SN + SP) / 2
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)
    Gmean = np.sqrt(SN * SP)
    F1 = 2 * Pre * SN / (Pre + SN)

    print(
        "Model score --- SN:{0:.4f} SP:{1:.4f} ACC:{2:.4f} BA:{3:.4f} MCC:{4:.4f} Pre:{5:.4f} AUC:{6:.4f} G-mean:{7:.4f} F1-score:{8:.4f}".format(
            SN, SP, ACC, BA, MCC, Pre, AUC, Gmean, F1))


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    run_model()
