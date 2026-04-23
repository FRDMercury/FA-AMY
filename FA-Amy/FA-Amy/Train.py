import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ======================================================
# CONFIGURATION AREA - PLEASE MODIFY THE PATHS BELOW
# ======================================================
# 1. Path to pre-computed training embeddings (.npy)
POS_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Generalized_dataset", "esmc_pos_train.npy")
NEG_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Generalized_dataset", "esmc_neg_train.npy")

# 2. Directory to save the trained models
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "Model-saved")

# 3. Training Hyperparameters
EPOCHS = 100
PATIENCE = 10         # Early stopping patience
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WARMUP_EPOCHS = 15    # Start saving model only after these epochs
SEED = 777

# 4. Loss Weighting
# Increase this value if you have fewer positive samples. 
# Example: 2.0 means positive samples are twice as important as negative ones.
POS_WEIGHT = 2.0
# ======================================================

def set_seed(seed=777):
    """Ensure high reproducibility for scientific research."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(SEED)

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
            layers.append(ResidualBlock(input_dim if i == 0 else num_channels[i-1], out_channels, kernel_size, dilation))
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

# ================== Training Process ==================

def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- FA-Amy Training (5-Fold CV) ---")
    print(f"Device: {device}")
    weight_tensor = torch.tensor([POS_WEIGHT]).to(device)
    # 1. Prepare Dataset
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    neg_train = np.load(NEG_TRAIN_PATH)
    pos_train = np.load(POS_TRAIN_PATH)
    
    df_prot = np.concatenate((pos_train, neg_train), axis=0)
    df_lb = np.concatenate((np.ones(len(pos_train), dtype=int), np.zeros(len(neg_train), dtype=int)))
    dataset = BioinformaticsDataset(df_lb, df_prot)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    all_metrics = {m: [] for m in ["SN", "SP", "ACC", "BA", "MCC", "Pre", "Gmean", "F1", "AUROC"]}
    fold_best_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_prot, df_lb)):
        print(f"\n>>> Fold {fold + 1} starting...")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
        
        model = FAAmyModule().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_ba = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            for prot_x, data_y in train_loader:
                prot_x, data_y = prot_x.to(device), data_y.to(device)
                optimizer.zero_grad()
                y_pred = model(prot_x)
                loss = F.binary_cross_entropy_with_logits(y_pred.view(-1), data_y, pos_weight = weight_tensor )
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for prot_x, data_y in val_loader:
                    prot_x = prot_x.to(device)
                    y_prob = torch.sigmoid(model(prot_x))
                    all_preds.extend(y_prob.cpu().numpy()); all_labels.extend(data_y.cpu().numpy())

            labels, probs = np.array(all_labels), np.array(all_preds)
            preds = np.around(probs)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            sn = tp / (tp + fn) if (tp + fn) > 0 else 0
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0
            ba = (sn + sp) / 2

            if ba > best_val_ba:
                if epoch >= WARMUP_EPOCHS:
                    best_val_ba = ba
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"Epoch {epoch+1:03d}: Val BA improved to {ba:.4f}. Model cached.")
                else:
                    print(f"Epoch {epoch+1:03d}: Val BA improved to {ba:.4f} (Warm-up).")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping triggered at Epoch {epoch+1}")
                    break

        # Finalize Fold
        best_path = os.path.join(MODEL_SAVE_DIR, f"best_fold{fold}.pth")
        torch.save(best_model_state, best_path)
        fold_best_models.append(best_path)

        # Calculate Final Metrics for Fold
        model.load_state_dict(best_model_state)
        model.eval()
        # [Metric Calculation Logic similar to your original code...]
        # (Snippet shortened for brevity, but all calculations like SN, SP, MCC, AUROC are included here)
        # Using the same logic as Test.py to ensure consistency
        y_true, y_probs = [], []
        with torch.no_grad():
            for prot_x, data_y in val_loader:
                y_prob = torch.sigmoid(model(prot_x.to(device)))
                y_true.extend(data_y.numpy()); y_probs.extend(y_prob.cpu().numpy())
        
        l, p = np.array(y_true), np.array(y_probs).flatten()
        pr = np.around(p)
        auroc = roc_auc_score(l, p)
        tn, fp, fn, tp = confusion_matrix(l, pr).ravel()
        sn = tp/(tp+fn); sp = tn/(tn+fp); acc = (tp+tn)/len(l); ba = (sn+sp)/2
        pre = tp/(tp+fp) if (tp+fp)>0 else 0; gmean = np.sqrt(sn*sp); f1 = 2*pre*sn/(pre+sn) if (pre+sn)>0 else 0
        mcc_den = np.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))
        mcc = ((tp*tn)-(fp*fn))/mcc_den if mcc_den>0 else 0

        print(f"Fold {fold+1} Metrics --- SN:{sn:.3f} SP:{sp:.3f} ACC:{acc:.3f} BA:{ba:.3f} AUROC:{auroc:.3f}")
        for k, v in zip(["SN","SP","ACC","BA","MCC","Pre","Gmean","F1","AUROC"], [sn,sp,acc,ba,mcc,pre,gmean,f1,auroc]):
            all_metrics[k].append(v)

    # Summary
    print("\n" + "="*20 + " CROSS VALIDATION SUMMARY " + "="*20)
    for m in all_metrics:
        print(f"{m:5s}: {np.mean(all_metrics[m]):.4f} ± {np.std(all_metrics[m]):.4f}")

    # Save final best model (from the best fold)
    best_fold_idx = np.argmax(all_metrics["BA"])
    final_path = os.path.join(MODEL_SAVE_DIR, "final_best_model.pth")
    torch.save(torch.load(fold_best_models[best_fold_idx]), final_path)
    print(f"\nFinal best model saved to: {final_path} (from Fold {best_fold_idx+1})")

if __name__ == "__main__":
    run_train()
