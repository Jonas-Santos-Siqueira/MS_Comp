import numpy as np
from dataclasses import dataclass
from typing import Optional
from .mscomp import MSComp, MSCompResults

def estimate_factor_pca(Y, n_factors=1, standardize=True):
    """
    Simple static PCA via SVD to extract common factor(s).
    Y: array (T x N) with no missing values (pre-impute if needed).
    Returns: (factors, loadings, explained_var_ratio)
    """
    Y = np.asarray(Y, dtype=float)
    if standardize:
        mu = Y.mean(axis=0, keepdims=True)
        sd = Y.std(axis=0, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (Y - mu) / sd
    else:
        mu = Y.mean(axis=0, keepdims=True)
        Z = Y - mu
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    # principal components: PCs = U * s
    PCs = U * s
    factors = PCs[:, :n_factors]
    loadings = Vt[:n_factors, :].T
    ev = (s**2) / (Z.shape[0] - 1)
    explained_ratio = ev / ev.sum()
    return factors, loadings, explained_ratio[:n_factors]

@dataclass
class FactorMSCompResults:
    factor_results: MSCompResults
    factors: np.ndarray
    loadings: np.ndarray
    explained_var_ratio: np.ndarray

class FactorMSComp:
    """
    Wrapper: estimate a common factor from a panel Y_t (T x N) via PCA,
    then fit an MSComp to that factor. This is a pragmatic "MS-Comp de fator".
    """
    def __init__(self, k_mean=4, k_var=2, ar_order=2, triangular_Pm=False, absorbing_last_mean=False,
                 share_ar_across_regimes=True, random_state: Optional[int]=None, n_factors:int=1, standardize:bool=True):
        self.n_factors = n_factors
        self.standardize = standardize
        self.inner = MSComp(k_mean=k_mean, k_var=k_var, ar_order=ar_order,
                            triangular_Pm=triangular_Pm, absorbing_last_mean=absorbing_last_mean,
                            share_ar_across_regimes=share_ar_across_regimes, random_state=random_state)

    def fit(self, Y):
        factors, loadings, evr = estimate_factor_pca(Y, n_factors=self.n_factors, standardize=self.standardize)
        # use the first factor by default
        f = factors[:, 0]
        res = self.inner.fit(f)
        return FactorMSCompResults(
            factor_results=res,
            factors=factors,
            loadings=loadings,
            explained_var_ratio=evr
        )

    def summary(self):
        return self.inner.summary()