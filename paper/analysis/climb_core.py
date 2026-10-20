"""Self-contained CLIMB replay on paper/data/records.csv: the threshold policy, the betting test on
the signed loss, fixed-sequence calibration, and exact risk on a finite pool. Vectorised over the
threshold grid; matches analysis/climb_calib.calibrate (checked in guarantee_iid.py)."""
import numpy as np, pandas as pd, os

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 0.6
GRID = [-0.005 * i for i in range(1, 121)] + [-0.6 - 0.05 * i for i in range(1, 49)] + [-1e9]
SETTINGS = [(m, b) for m in ('GLM', 'DeepSeek') for b in ('MATH', 'GAIA', 'FRAMES', 'MuSiQue')]

def load(path=os.path.join(HERE, '..', 'data', 'records.csv')):
    return pd.read_csv(path)

def outcome_matrix(d):
    """Rows: candidates (direct, then GRID in testing order); columns: tasks. Returns Y, C, loss."""
    S0, H0, S1, H1 = (d[k].to_numpy(float) for k in ('S0', 'H0', 'S1', 'H1'))
    Y = [d['YK'].to_numpy(int)]; C = [d['CK'].to_numpy(float)]
    for t in GRID:
        s0 = (S0 > t) & (H0 <= TAU); s1 = ~s0 & (S1 > t) & (H1 <= TAU)
        Y.append(np.where(s0, d['Y0'], np.where(s1, d['Y1'], d['Y2'])))
        C.append(np.where(s0, d['C0'], np.where(s1, d['C1'], d['C2'])))
    Y, C = np.array(Y, int), np.array(C, float)
    return Y, C, d['YK'].to_numpy(int)[None, :] - Y

def p_betting(L, eps, delta):
    """Betting p-values for each row of the loss matrix L (candidates x calibration order)."""
    n = L.shape[1]; t = np.arange(1, n + 1)
    s1 = np.cumsum(L, 1) - L; s2 = np.cumsum(L * L, 1) - L * L          # sums over earlier samples
    var = np.maximum((s2 + 0.25) / t - (s1 / t) ** 2, 1e-3)
    lam = np.minimum(0.5 / (1 - eps), np.sqrt(2 * np.log(1 / delta) / (var * n)))
    K = np.cumprod(1 + lam * (eps - L), 1)
    return 1.0 / np.maximum(K.max(1), 1.0)

def calibrate(L_cal, C_cal, eps, delta=0.1):
    """Index of the deployed candidate (0 = run the pipeline)."""
    p = p_betting(L_cal[1:], eps, delta)
    fail = np.nonzero(p > delta)[0]
    m = fail[0] if len(fail) else len(p)                                 # certified: GRID[:m]
    return int(np.argmin(C_cal[:m + 1].mean(1)))
