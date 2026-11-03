"""
Heavier-model test: a small Transformer encoder (an LSTM was tried first, see git history of
this file's predecessor run) trained directly on 60-day windows of raw daily log-returns,
retrained periodically on an expanding walk-forward window, to classify 3-day-ahead direction.
Class-balanced loss (pos_weight from the training window's own class ratio) to rule out the
model just learning the label imbalance.

Result (see README Finding 11): both the LSTM and this Transformer collapse to near-constant
output (`prob_std` as low as 0.006-0.011) and land AT OR BELOW the trivial "always predict the
majority class" baseline on every asset. More model capacity did not help, because the problem
was never capacity -- every other experiment in this directory (shallow trees, Ridge, logistic
regression) already showed the available features/raw returns don't carry much learnable
signal beyond what karcher's simple historical average already captures. A bigger model trained
on the same thin signal just overfits noise or collapses to the prior.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from data import download_close_prices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 60
HORIZON = 3

class TransformerDirClassifier(nn.Module):
    def __init__(self, d_model=32, nhead=4, layers=2, in_dim=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, W, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64,
                                                dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        h = self.proj(x) + self.pos
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)

def build_sequences(log_ret, horizon=HORIZON, w=W):
    n = len(log_ret)
    X, y, origin_idx = [], [], []
    for t in range(w, n - horizon):
        X.append(log_ret[t - w: t]); y.append(1.0 if log_ret[t:t+horizon].sum() > 0 else 0.0); origin_idx.append(t)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(origin_idx)

def train_model(X_train, y_train, epochs=25, lr=1e-3, batch=64, pos_weight=None):
    model = TransformerDirClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt = torch.tensor(X_train).unsqueeze(-1).to(device); yt = torch.tensor(y_train).to(device)
    n = len(Xt); model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad(); loss = lossf(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    model.eval(); return model

def predict(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X).unsqueeze(-1).to(device))).cpu().numpy()

def run(ticker, name, test_step=3, retrain_every=10, min_train_days=300, epochs=25):
    t0 = time.time()
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices)).astype(np.float32)
    X, y, origin_idx = build_sequences(log_ret)
    n = len(X)
    start_i = max(np.searchsorted(origin_idx, min_train_days), 1)
    test_points = list(range(start_i, n, test_step))
    model, preds, actuals, probs = None, [], [], []
    for k, i in enumerate(test_points):
        if model is None or k % retrain_every == 0:
            ytr = y[:i]
            pw = torch.tensor([ (len(ytr) - ytr.sum()) / max(ytr.sum(), 1) ]).to(device)  # balance classes
            model = train_model(X[:i], ytr, epochs=epochs, pos_weight=pw)
        p_up = float(predict(model, X[i:i+1])[0])
        probs.append(p_up)
        preds.append(1 if p_up > 0.5 else 0); actuals.append(int(y[i]))
    preds = np.array(preds); actuals = np.array(actuals); probs = np.array(probs)
    naive_acc = float((actuals == 0).mean())
    model_acc = float((preds == actuals).mean())
    print(f"{name:10s} n={len(preds):3d}  Transformer dir_acc={model_acc:.3f}  "
          f"pred_up_rate={preds.mean():.3f}  prob_std={probs.std():.3f}  "
          f"always-down={naive_acc:.3f}  always-up={1-naive_acc:.3f}  ({time.time()-t0:.0f}s)")

for ticker, name in [("BZ=F","Brent"), ("^GSPC","S&P500"), ("BTC-USD","Bitcoin")]:
    run(ticker, name)
