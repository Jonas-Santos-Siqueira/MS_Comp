import numpy as np
from mscomp import MSComp, FactorMSComp

np.random.seed(0)
T = 400
# True regimes: switch mean every 100, variance every 50
mu = np.array([0.0, 1.0, -0.5, 0.8])
sigma = np.array([0.5, 1.5])
mean_regime = np.repeat([0,1,2,3], 100)
var_regime = np.tile(np.repeat([0,1], 50), 4)
phi = np.array([0.3, 0.2])
y = np.zeros(T)
for t in range(T):
    ar = (phi[0]*y[t-1] if t-1>=0 else 0.0) + (phi[1]*y[t-2] if t-2>=0 else 0.0)
    eps = np.random.normal(0, sigma[var_regime[t]])
    y[t] = mu[mean_regime[t]] + ar + eps

model = MSComp(k_mean=4, k_var=2, ar_order=2, absorbing_last_mean=False)
res = model.fit(y, maxiter=100, tol=1e-5, verbose=False)
print(model.summary())

# Factor demo: create 5-series panel loaded on y + noise
Y = np.vstack([y + np.random.normal(scale=0.3, size=T) for _ in range(5)]).T
fmodel = FactorMSComp(k_mean=4, k_var=2, ar_order=2)
fres = fmodel.fit(Y)
print(fmodel.summary())