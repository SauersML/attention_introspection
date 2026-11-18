# Shared analysis helpers (run on the pod or anywhere with torch+numpy).
import numpy as np, torch
BAND = list(range(21, 54))

def pertok(a, ntok):
    """a: [NL,H,ns] sums -> per-token, layer band mean over heads -> [ns]"""
    a = a.float()[BAND].mean(0).mean(0)  # [ns]
    return (a / torch.tensor(ntok[:a.shape[0]], dtype=torch.float)).numpy()

def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def signp(x, B=20000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], (B, len(x))) * x[None]
    return float((np.abs(s.mean(1)) >= abs(x.mean())).mean())

def pct(lr):  # mean log ratio -> percent with CI
    m, lo, hi = boot(lr)
    return dict(pct=100 * (np.exp(m) - 1), lo=100 * (np.exp(lo) - 1), hi=100 * (np.exp(hi) - 1), p=signp(lr))
