# mscomp (Python)

Markov-switching **mean–variance component** model (MS-Comp) with **independent chains** for the mean and the variance:

\[
y_t = a_{S^m_t} + \sum_{\ell=1}^p \phi_\ell y_{t-\ell} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2_{S^v_t})
\]

- Two independent Markov chains with transitions \(P_m\) and \(P_v\); total transition \(P = P_v \otimes P_m\).
- Options for **upper-triangular \(P_m\)** (no backward moves) and **absorbing last mean regime** (permanent break).
- Estimation by direct MLE using the Hamilton filter on the combined state-space and a simple smoother.
- Supports AR(p) in the conditional mean (MA(q) can be added via a state-space extension).

## Quick start

```python
import numpy as np
from mscomp import MSComp, FactorMSComp

# Univariate MS-Comp on a series y
y = np.loadtxt("your_series.csv")
model = MSComp(k_mean=4, k_var=2, ar_order=2, triangular_Pm=False, absorbing_last_mean=False)
res = model.fit(y)
print(model.summary())

# Factor MS-Comp: extract a common factor from a panel Y (T x N), then fit MS-Comp on it
Y = np.loadtxt("your_panel.csv", delimiter=",")
fmodel = FactorMSComp(k_mean=4, k_var=2, ar_order=2, absorbing_last_mean=True)
fres = fmodel.fit(Y)
print(fmodel.summary())
```

### Outputs
- `res.mean_regime_probs` and `res.var_regime_probs`: smoothed probabilities (T x k_mean / k_var).
- `res.fitted`: one-step ahead fitted values; `res.residuals`: prediction errors.
- `res.Pm`, `res.Pv`: estimated transition matrices.

### Notes
- This implementation currently supports **AR(p)** in the mean. Adding MA(q) is straightforward via a state-space representation (Kalman) and is left as a planned extension.
- The "Factor MS-Comp" approach here is **pragmatic**: first extract a factor (PCA) and then apply MS-Comp to that factor. A fully joint **dynamic factor + MS-Comp** model is a heavier extension and can be added later.
```
