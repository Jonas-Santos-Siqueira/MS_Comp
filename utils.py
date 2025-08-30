import numpy as np

def row_softmax(theta_row):
    # stable softmax for a 1D row
    m = np.max(theta_row)
    e = np.exp(theta_row - m)
    return e / e.sum()

def build_transition(theta, mask=None, absorbing_last=False, triangular=False):
    """
    Map unconstrained square matrix theta -> row-stochastic transition matrix.
    Optionally impose:
        - absorbing_last: last state is absorbing (last row is [0,...,0,1])
        - triangular: zero out strictly lower triangle (no backward transitions), renormalize each row
    mask: boolean matrix where False entries are forced to zero before row-normalization.
    """
    k = theta.shape[0]
    P = np.zeros_like(theta)
    for i in range(k):
        row = theta[i].copy()
        if triangular:
            # zero entries j < i (strictly lower triangle) by setting -inf before softmax
            for j in range(k):
                if j < i:
                    row[j] = -1e9
        if mask is not None:
            # where mask[i,j] == False, forbid transition
            for j in range(k):
                if not mask[i, j]:
                    row[j] = -1e9
        if absorbing_last and i == k - 1:
            P[i] = np.eye(k)[-1]
        else:
            P[i] = row_softmax(row)
    return P

def logsumexp(logw):
    m = np.max(logw)
    return m + np.log(np.sum(np.exp(logw - m)))

def kron(Pv, Pm):
    return np.kron(Pv, Pm)

def stable_cholesky(var):
    # for scalar variance; ensure positivity
    return np.sqrt(max(var, 1e-12))

def to_positive(x):
    # softplus for strictly positive parameters (variance)
    return np.log1p(np.exp(x)) + 1e-6