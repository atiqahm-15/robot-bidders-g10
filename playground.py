import random
import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# Set random seed for reproducibility
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report, 
    confusion_matrix, precision_score, recall_score, f1_score
)
import pandas as pd
 
from mlp_branch import StaticMLPBranch
from lstm import lstm

# =====================================================================
# 1. Model Definitions (Step 6)
# =====================================================================

class MLPClassifier(nn.Module):
    def __init__(self, static_input_dim=20, embedding_dim=32, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        self.mlp_branch = StaticMLPBranch(input_dim=static_input_dim,
                                          embedding_dim=embedding_dim,
                                          dropout_rate=dropout_rate)
        # Final dense layers to output logits (sigmoid applied at inference)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_static, x_seq):
        # Ignores sequential data
        emb = self.mlp_branch(x_static)
        return self.classifier(emb).squeeze(1)

class LSTMClassifier(nn.Module):
    def __init__(self, seq_input_dim=3, embedding_dim=32, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = lstm(seq_input_dim=seq_input_dim,
                         embedding_dim=embedding_dim,
                         dropout_rate=dropout_rate)
        # Final dense layers to output logits (sigmoid applied at inference)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_static, x_seq):
        # Ignores static data
        emb = self.lstm(x_seq)
        return self.classifier(emb).squeeze(1)

class HybridBotDetector(nn.Module):
    def __init__(self, static_input_dim=20, seq_input_dim=3, embedding_dim=32, dropout_rate=0.3):
        super(HybridBotDetector, self).__init__()
 
        self.mlp_branch  = StaticMLPBranch(input_dim=static_input_dim,
                                            embedding_dim=embedding_dim,
                                            dropout_rate=dropout_rate)
 
        self.lstm = lstm(seq_input_dim=seq_input_dim,
                                       embedding_dim=embedding_dim,
                                       dropout_rate=dropout_rate)
 
        fused_dim = embedding_dim * 2  # 32 + 32 = 64
        # Final dense layers to output logits (sigmoid applied at inference)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)  
        )
 
    def forward(self, x_static, x_seq):
        mlp_out  = self.mlp_branch(x_static)             # (batch, 32)
        lstm_out = self.lstm(x_seq)                      # (batch, 32)
        # Concatenate embeddings
        fused    = torch.cat([mlp_out, lstm_out], dim=1) # (batch, 64)
        return self.fusion(fused).squeeze(1)             # (batch,)
 

# =====================================================================
# 2. Data Loading & Preparation
# =====================================================================

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3

print("Loading data...")
X_static_train = np.load('npy_var/X_static_train.npy')
X_seq_train    = np.load('npy_var/X_seq_train.npy')
y_train        = np.load('npy_var/y_train.npy')
X_static_test  = np.load('npy_var/X_static_test.npy')
X_seq_test     = np.load('npy_var/X_seq_test.npy')
 
# split
idx_train, idx_val = train_test_split(
    np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=42
)
 
def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype)
 
# train tensor
static_tr = to_tensor(X_static_train[idx_train])
seq_tr    = to_tensor(X_seq_train[idx_train])
y_tr      = to_tensor(y_train[idx_train])
 
# val tensor
static_val = to_tensor(X_static_train[idx_val])
seq_val    = to_tensor(X_seq_train[idx_val])
y_val      = to_tensor(y_train[idx_val])
 
# Oversample the minority class (robots ~5%) during training
class_counts = np.bincount(y_train[idx_train].astype(int))
pos_weight = torch.tensor([np.sqrt(class_counts[0] / class_counts[1])], dtype=torch.float32).to(DEVICE) # softer 
 
train_ds  = TensorDataset(static_tr, seq_tr, y_tr)
val_ds    = TensorDataset(static_val, seq_val, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
 

# =====================================================================
# 3. Evaluation Pipelines (Step 7)
# =====================================================================

def cost_sensitive_evaluation(y_true, y_pred_binary, cost_fp=10, cost_fn=1):
    """
    Cost-sensitive evaluation.
    Misclassifying a legitimate user as a bot (FP) is generally more costly 
    than failing to detect a bot (FN) due to UX impact.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return total_cost

def evaluate_model(model, loader):
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for x_s, x_q, y_b in loader:
            x_s, x_q = x_s.to(DEVICE), x_q.to(DEVICE)
            # Sigmoid output layer for prediction probability
            probs = torch.sigmoid(model(x_s, x_q)).cpu().numpy()
            val_preds.extend(probs)
            val_labels.extend(y_b.numpy())
            
    # Main metrics
    auc_score = roc_auc_score(val_labels, val_preds)
    pr_auc = average_precision_score(val_labels, val_preds)
    
    # Threshold tuning based on F1-score
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds_bin = [1 if p >= t else 0 for p in val_preds]
        f1 = f1_score(val_labels, preds_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    # Calculate metrics at the best threshold
    preds_bin = [1 if p >= best_thresh else 0 for p in val_preds]
    cm = confusion_matrix(val_labels, preds_bin)
    precision = precision_score(val_labels, preds_bin, zero_division=0)
    recall = recall_score(val_labels, preds_bin, zero_division=0)
    
    # Cost-sensitive evaluation: Cost FP = 10, Cost FN = 1
    cost = cost_sensitive_evaluation(val_labels, preds_bin, cost_fp=10, cost_fn=1)
    
    return {
        "roc_auc": auc_score,
        "pr_auc": pr_auc,
        "best_thresh": best_thresh,
        "f1": best_f1,
        "precision": precision,
        "recall": recall,
        "cm": cm,
        "cost": cost,
        "val_preds": val_preds,
        "val_labels": val_labels
    }

def train_and_evaluate(model, name):
    print(f"\n{'='*50}\nTraining {name} Model\n{'='*50}")
    model = model.to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_auc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x_s, x_q, y_b in train_loader:
            x_s, x_q, y_b = x_s.to(DEVICE), x_q.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_s, x_q)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent LSTM exploding gradients
            optimizer.step()
            train_loss += loss.item()
            
        # Validation to track best model
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_s, x_q, y_b in val_loader:
                x_s, x_q = x_s.to(DEVICE), x_q.to(DEVICE)
                probs = torch.sigmoid(model(x_s, x_q)).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(y_b.numpy())
                
        auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(1 - auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f'best_{name.lower()}.pt')
            
    print(f"Finished training {name}. Best Val ROC-AUC: {best_auc:.4f}")
    
    # Load best model for final rigorous evaluation
    model.load_state_dict(torch.load(f'best_{name.lower()}.pt'))
    results = evaluate_model(model, val_loader)
    
    print(f"\n--- Evaluation Results for {name} ---")
    print(f"ROC AUC:           {results['roc_auc']:.4f}")
    print(f"PR AUC:            {results['pr_auc']:.4f}")
    print(f"Best Threshold:    {results['best_thresh']:.2f} (Tuned for Max F1)")
    print(f"F1-Score:          {results['f1']:.4f}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    print(f"Cost (FP=10,FN=1): {results['cost']}")
    print(f"Confusion Matrix:\n{results['cm']}")
    
    return results

# =====================================================================
# 4. Execute Full Comparison
# =====================================================================

if __name__ == "__main__":
    models_to_test = {
        "MLP": MLPClassifier(),
        "LSTM": LSTMClassifier(),
        "Hybrid": HybridBotDetector()
    }

    all_results = {}
    for name, m in models_to_test.items():
        all_results[name] = train_and_evaluate(m, name)

    # Summary Comparison Table
    print(f"\n\n{'='*65}\nFINAL COMPARISON SUMMARY\n{'='*65}")
    print(f"{'Model':<10} | {'ROC AUC':<8} | {'PR AUC':<8} | {'F1-Score':<8} | {'Cost':<6}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<10} | {res['roc_auc']:.4f}   | {res['pr_auc']:.4f}   | {res['f1']:.4f}   | {res['cost']}")
        
    print("\nAs shown above, comparing the Hybrid model against individual MLP and LSTM branches")
    print("allows us to see the exact benefits of fusing Static and Sequential features.")

    # Generate Test Predictions using the Hybrid Model
    print("\nGenerating final test set predictions with Hybrid model...")
    hybrid_model = models_to_test["Hybrid"]
    hybrid_model.load_state_dict(torch.load('best_hybrid.pt'))
    hybrid_model.eval()
    
    static_test = to_tensor(X_static_test)
    seq_test    = to_tensor(X_seq_test)
    test_ds     = TensorDataset(static_test, seq_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    test_preds = []
    with torch.no_grad():
        for x_s, x_q in test_loader:
            x_s, x_q = x_s.to(DEVICE), x_q.to(DEVICE)
            probs = torch.sigmoid(hybrid_model(x_s, x_q)).cpu().numpy()
            test_preds.extend(probs)
            
    try:
        import os
        os.makedirs('results', exist_ok=True)
        test = pd.read_csv('data/test.csv')                       
        test['prediction'] = test_preds
        test[['bidder_id', 'prediction']].to_csv('results/submission.csv', index=False)
        print("Successfully generated test predictions and saved to results/submission.csv.")
    except FileNotFoundError:
        print("data/test.csv not found locally, skipping submission file generation.")