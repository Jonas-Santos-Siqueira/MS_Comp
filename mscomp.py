import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from .utils import build_transition, kron, logsumexp, to_positive

@dataclass
class MSCompResults:
    params: Dict[str, np.ndarray]
    Pm: np.ndarray
    Pv: np.ndarray
    P: np.ndarray
    loglike: float
    filtered_probs: np.ndarray  # T x (km*kv)
    smoothed_probs: np.ndarray  # T x (km*kv)
    mean_regime_probs: np.ndarray  # T x km (smoothed marginal)
    var_regime_probs: np.ndarray   # T x kv (smoothed marginal)
    fitted: np.ndarray            # in-sample one-step-ahead predictions
    residuals: np.ndarray
    ar_order: int
    k_mean: int
    k_var: int

class MSComp:
    """
    Markov-switching mean–variance component model (Gaussian), univariate:
      y_t = a_{S^m_t} + sum_{l=1..p} phi_l y_{t-l} + eps_t,   eps_t ~ N(0, sigma^2_{S^v_t})
    with independent Markov chains for mean (S^m) and variance (S^v),
    transition matrices Pm and Pv, and total transition P = kron(Pv, Pm).
    
    Parameters
    ----------
    k_mean : int
        Number of mean regimes.
    k_var : int
        Number of variance regimes.
    ar_order : int
        AR order (integer >= 0).
    triangular_Pm : bool
        If True, impose upper-triangular transitions on Pm (no backward moves).
    absorbing_last_mean : bool
        If True, last mean regime is absorbing (permanent break).
    share_ar_across_regimes : bool
        If True, AR coefficients are shared across mean regimes (recommended).
    """

    def __init__(self, k_mean=4, k_var=2, ar_order=2, triangular_Pm=False, absorbing_last_mean=False,
                 share_ar_across_regimes=True, random_state: Optional[int]=None):
        self.km = k_mean
        self.kv = k_var
        self.p = ar_order
        self.triangular_Pm = triangular_Pm
        self.absorbing_last_mean = absorbing_last_mean
        self.share_ar = share_ar_across_regimes
        self.rng = np.random.default_rng(random_state)
        self.fitted_ = None

    # ---------- Parameter packing / unpacking ----------
    def _pack(self, a, phi, log_sigmasq, theta_m, theta_v):
        return np.concatenate([a.ravel(), phi.ravel(), log_sigmasq.ravel(),
                               theta_m.ravel(), theta_v.ravel()])

    def _unpack(self, params):
        idx = 0
        a = params[idx: idx + self.km]; idx += self.km
        phi = params[idx: idx + (self.p if self.share_ar else self.km * self.p)]; idx += (self.p if self.share_ar else self.km * self.p)
        log_sigmasq = params[idx: idx + self.kv]; idx += self.kv
        theta_m = params[idx: idx + self.km*self.km].reshape(self.km, self.km); idx += self.km*self.km
        theta_v = params[idx: idx + self.kv*self.kv].reshape(self.kv, self.kv); idx += self.kv*self.kv
        return a, phi, log_sigmasq, theta_m, theta_v

    def _init_params(self, y):
        y = np.asarray(y).astype(float)
        T = len(y)
        # intercepts initialized around mean splits
        quantiles = np.quantile(y, np.linspace(0.1, 0.9, self.km))
        a0 = np.linspace(quantiles[0], quantiles[-1], self.km)
        # AR: Yule-Walker (rough) for AR(p); fallback zeros
        phi0 = np.zeros(self.p if self.share_ar else self.km * self.p)
        # variance regimes: split by quantiles of residuals w.r.t. mean
        s2 = np.var(y - np.mean(y))
        sig0 = np.log(np.linspace(0.5*s2, 1.5*s2, self.kv))
        # transitions: start near diagonal
        theta_m0 = np.full((self.km, self.km), -2.0)
        np.fill_diagonal(theta_m0, 2.0)
        theta_v0 = np.full((self.kv, self.kv), -2.0)
        np.fill_diagonal(theta_v0, 2.0)
        return self._pack(a0, phi0, sig0, theta_m0, theta_v0)

    def _emission_ll(self, y_t, y_hist, a, phi, sigmasq):
        # compute emission log-likelihood for each mean/var regime combination
        # y_hist: vector [y_{t-1},...,y_{t-p}]
        km, kv = self.km, self.kv
        if self.share_ar:
            mu_by_mean = a + (phi @ y_hist) if self.p > 0 else a.copy()
        else:
            mu_by_mean = np.zeros(km)
            for i in range(km):
                start = i*self.p
                phi_i = phi[start:start+self.p]
                mu_by_mean[i] = a[i] + (phi_i @ y_hist if self.p > 0 else 0.0)
        # Build combined grid of means/variances
        ll = np.empty(km*kv)
        idx = 0
        for iv in range(kv):
            var = sigmasq[iv]
            inv = 1.0 / var
            c = -0.5*np.log(2*np.pi*var)
            for im in range(km):
                err = y_t - mu_by_mean[im]
                ll[idx] = c - 0.5*err*err*inv
                idx += 1
        return ll  # length km*kv

    def _filter(self, y, Pm, Pv, a, phi, sigmasq) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Hamilton filter for combined states with P = kron(Pv, Pm)
        Returns:
          loglike, filtered_probs (T x S), fitted (T,)
        """
        y = np.asarray(y).astype(float)
        T = len(y)
        S = self.km * self.kv
        P = kron(Pv, Pm)
        # initial state prob: stationary of P (fallback uniform)
        try:
            w, v = np.linalg.eig(P.T)
            idx = np.argmin(np.abs(w - 1.0))
            pi = np.real(v[:, idx]); pi = pi / pi.sum(); pi = np.maximum(pi, 1e-12); pi = pi / pi.sum()
        except Exception:
            pi = np.ones(S)/S

        loglike = 0.0
        filtered = np.zeros((T, S))
        fitted = np.full(T, np.nan)
        # start after p lags
        start = self.p
        # simple fallback for initial AR history: use sample mean for missing lags
        ypad = np.concatenate([np.repeat(y[:1].mean(), self.p), y])
        alpha_prev = pi.copy()
        for t in range(T):
            if t < start:
                # use mean-only until enough lags
                y_hist = np.zeros(self.p)
            else:
                y_hist = ypad[t + self.p - self.p: t + self.p][::-1]  # last p values reversed
            # Predict step
            alpha_pred = P.T @ alpha_prev  # shape (S,)
            # Emission
            ll = self._emission_ll(y[t], y_hist, a, phi, sigmasq)  # shape (S,)
            # Update (in log-space)
            logw = np.log(alpha_pred + 1e-300) + ll
            logc = logsumexp(logw)
            alpha = np.exp(logw - logc)
            loglike += logc
            filtered[t] = alpha
            # one-step-ahead fitted value: E[y_t | info_{t-1}] = sum_s E[y_t|s] * alpha_pred_s
            # approximate using means at y_hist
            if self.share_ar:
                mu_by_mean = a + (phi @ y_hist) if self.p > 0 else a.copy()
            else:
                mu_by_mean = np.zeros(self.km)
                for im in range(self.km):
                    starti = im*self.p
                    phi_i = phi[starti:starti+self.p]
                    mu_by_mean[im] = a[im] + (phi_i @ y_hist if self.p > 0 else 0.0)
            # combine over variance states
            mu_full = np.repeat(mu_by_mean, self.kv)
            fitted[t] = np.dot(mu_full, alpha_pred)
            alpha_prev = alpha
        return loglike, filtered, fitted

    def _smooth(self, filtered, Pm, Pv) -> np.ndarray:
        T, S = filtered.shape
        P = kron(Pv, Pm)
        smoothed = np.zeros_like(filtered)
        smoothed[-1] = filtered[-1]
        for t in range(T-2, -1, -1):
            # beta_t = P * (filtered_{t+1} ./ (P'.dot(filtered_t)))
            denom = P.T @ filtered[t]
            denom = np.where(denom < 1e-300, 1e-300, denom)
            beta = P @ (smoothed[t+1] / denom)
            smoothed[t] = filtered[t] * beta
            s = smoothed[t].sum()
            if s <= 0:
                smoothed[t] = filtered[t]
            else:
                smoothed[t] /= s
        return smoothed

    def _negloglike(self, params, y) -> float:
        a, phi, log_sigmasq, theta_m, theta_v = self._unpack(params)
        sigmasq = np.exp(log_sigmasq)  # positive
        Pm = build_transition(theta_m, absorbing_last=self.absorbing_last_mean, triangular=self.triangular_Pm)
        Pv = build_transition(theta_v, absorbing_last=False, triangular=False)
        ll, _, _ = self._filter(y, Pm, Pv, a, phi, sigmasq)
        return -ll

    def fit(self, y, maxiter=200, tol=1e-6, method="L-BFGS-B", verbose=False) -> MSCompResults:
        from scipy.optimize import minimize

        y = np.asarray(y).astype(float).ravel()
        x0 = self._init_params(y)
        obj = lambda th: self._negloglike(th, y)
        res = minimize(obj, x0, method=method, options={"maxiter": maxiter, "disp": verbose, "gtol": tol})
        a, phi, log_sigmasq, theta_m, theta_v = self._unpack(res.x)
        sigmasq = np.exp(log_sigmasq)
        Pm = build_transition(theta_m, absorbing_last=self.absorbing_last_mean, triangular=self.triangular_Pm)
        Pv = build_transition(theta_v, absorbing_last=False, triangular=False)
        ll, filt, fitted = self._filter(y, Pm, Pv, a, phi, sigmasq)
        smooth = self._smooth(filt, Pm, Pv)
        # marginals
        S = self.km * self.kv
        mean_probs = np.zeros((len(y), self.km))
        var_probs  = np.zeros((len(y), self.kv))
        for t in range(len(y)):
            st = smooth[t]
            # reshape to (km, kv)
            M = st.reshape(self.km, self.kv)
            mean_probs[t] = M.sum(axis=1)
            var_probs[t]  = M.sum(axis=0)
        resid = y - fitted
        params = {
            "a": a, "phi": phi, "sigma2": sigmasq,
            "theta_m": theta_m, "theta_v": theta_v
        }
        self.fitted_ = MSCompResults(
            params=params, Pm=Pm, Pv=Pv, P=kron(Pv, Pm), loglike=ll,
            filtered_probs=filt, smoothed_probs=smooth,
            mean_regime_probs=mean_probs, var_regime_probs=var_probs,
            fitted=fitted, residuals=resid, ar_order=self.p,
            k_mean=self.km, k_var=self.kv
        )
        return self.fitted_

    def predict_in_sample(self):
        if self.fitted_ is None:
            raise RuntimeError("Call fit() first.")
        return self.fitted_.fitted

    def summary(self) -> str:
        if self.fitted_ is None:
            return "MSComp(not fitted)."
        r = self.fitted_
        lines = []
        lines.append("=== MSComp Results ===")
        lines.append(f"log-likelihood: {r.loglike:.3f}")
        lines.append(f"Mean regimes: {r.k_mean}, Variance regimes: {r.k_var}, AR order: {r.ar_order}")
        lines.append("Intercepts (a_i): " + ", ".join(f"{x:.4f}" for x in r.params["a"]))
        if self.share_ar:
            lines.append("Shared AR coefficients (phi): " + ", ".join(f"{x:.4f}" for x in r.params["phi"]))
        else:
            lines.append("AR coefficients by regime: " + ", ".join(f"{x:.4f}" for x in r.params["phi"]))
        lines.append("Sigma^2 (variance regimes): " + ", ".join(f"{x:.6f}" for x in r.params["sigma2"]))
        lines.append("Pm (mean transitions):\n" + np.array2string(r.Pm, formatter={'float_kind':lambda x: f'{x: .4f}'}))
        lines.append("Pv (variance transitions):\n" + np.array2string(r.Pv, formatter={'float_kind':lambda x: f'{x: .4f}'}))
        return "\n".join(lines)