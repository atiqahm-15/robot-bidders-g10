import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
 
from mlp_branch import StaticMLPBranch
from lstm import lstm
 
 
# combine mlp and lstm
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
        fused    = torch.cat([mlp_out, lstm_out], dim=1) # (batch, 64)
        return self.fusion(fused).squeeze(1)             # (batch,)
 
 
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3

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
weights      = 1.0 / class_counts[y_train[idx_train].astype(int)]
sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
 
train_ds  = TensorDataset(static_tr, seq_tr, y_tr)
val_ds    = TensorDataset(static_val, seq_val, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) # sampler=sampler for weighted random sampling
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
 
model     = HybridBotDetector().to(DEVICE)
# pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(DEVICE)
pos_weight = torch.tensor([np.sqrt(class_counts[0] / class_counts[1])], dtype=torch.float32).to(DEVICE) # softer 
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
 
best_f1 = 0.0
best_auc   = 0.0
 
#  train
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
 
    # val
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
 
    print(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")
 
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"           ↑ New best saved (AUC={auc:.4f})")
 
 

model.load_state_dict(torch.load('best_model.pt'))
model.eval()
val_preds, val_labels = [], []
with torch.no_grad():
    for x_s, x_q, y_b in val_loader:
        x_s, x_q = x_s.to(DEVICE), x_q.to(DEVICE)
        probs = torch.sigmoid(model(x_s, x_q)).cpu().numpy()
        val_preds.extend(probs)
        val_labels.extend(y_b.numpy())
 
print("\n=== Final Validation Results ===")
print(f"ROC-AUC: {roc_auc_score(val_labels, val_preds):.4f}")
print(classification_report(val_labels, [1 if p >= 0.5 else 0 for p in val_preds],
                             target_names=['Human', 'Robot']))
 
 
static_test = to_tensor(X_static_test)
seq_test    = to_tensor(X_seq_test)
test_ds     = TensorDataset(static_test, seq_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
 
model.eval()
test_preds = []
with torch.no_grad():
    for x_s, x_q in test_loader:
        x_s, x_q = x_s.to(DEVICE), x_q.to(DEVICE)
        probs = torch.sigmoid(model(x_s, x_q)).cpu().numpy()
        test_preds.extend(probs)
 

import pandas as pd
test = pd.read_csv('test.csv')                       
test['prediction'] = test_preds
# test[['bidder_id', 'prediction']].to_csv('submission.csv', index=False)


from sklearn.metrics import f1_score

thresholds = np.arange(0.3, 0.7, 0.01)
best_thresh = max(thresholds, key=lambda t: f1_score(val_labels, [1 if p >= t else 0 for p in val_preds]))
print(f"Best threshold: {best_thresh:.2f}")

# Use best_thresh for final evaluation and test predictions
print(classification_report(val_labels, [1 if p >= best_thresh else 0 for p in val_preds],
                             target_names=['Human', 'Robot']))


macro_f1 = f1_score(val_labels, [1 if p >= 0.5 else 0 for p in val_preds], average='macro')

if macro_f1 > best_f1:  # track best_f1 instead of best_auc
    best_f1 = macro_f1
    torch.save(model.state_dict(), 'best_model.pt')