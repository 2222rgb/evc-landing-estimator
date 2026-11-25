"""Constrained EVC landing-share estimator with shared delay curve, block bootstrap, and lead check."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from evc_landing_estimator.config import Config


@dataclass
class PreparedDesign:
    dates: pd.DatetimeIndex
    channels: list[str]
    Y: np.ndarray  # (T, C)
    E: np.ndarray  # (T,)
    E_lags: np.ndarray  # (T, Lmax+1), col L is E(t - L)
    z_lead: np.ndarray  # (T,), mean(E(t+1), E(t+2))
    D: np.ndarray  # (T, p), intercept + trend + 6 DOW + controls


@dataclass
class FitOutput:
    dates: pd.DatetimeIndex
    channels: list[str]
    shares: np.ndarray  # (C,)
    unconstrained_shares: np.ndarray  # (C,)
    ci_low: np.ndarray  # (C,)
    ci_high: np.ndarray  # (C,)
    p_top: np.ndarray  # (C,)
    p_material: np.ndarray  # (C,)
    partial_r2: np.ndarray  # (C,)
    lead_coef: np.ndarray  # (C,)
    lead_ci_low: np.ndarray  # (C,)
    lead_ci_high: np.ndarray  # (C,)
    lead_flag: np.ndarray  # (C,) bool
    not_visible: float
    not_visible_ci: tuple[float, float]
    k: np.ndarray  # (Lmax+1,)
    k_ci_low: np.ndarray  # (Lmax+1,)
    k_ci_high: np.ndarray  # (Lmax+1,)
    confounded: bool
    converged: bool
    baseline_fit: np.ndarray  # (T, C)
    evc_component: np.ndarray  # (T, C)
    evc_series: np.ndarray  # (T,)
    actual_y: np.ndarray  # (T, C)
    warnings: list[str] = field(default_factory=list)


def project_simplex_ball(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection onto {w >= 0, sum(w) <= 1} (Duchi et al. 2008)."""
    u = np.maximum(np.asarray(v, dtype=float), 0.0)
    if u.sum() <= 1.0:
        return u
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s) - 1.0
    ind = np.arange(1, len(u) + 1, dtype=float)
    cond = s - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(u, 1.0 / len(u))
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(u - theta, 0.0)


def build_design(
    conv_daily: pd.DataFrame,
    evc_daily: pd.Series,
    max_lag: int,
    controls: pd.DataFrame | None = None,
) -> PreparedDesign:
    """Align conversions, EVC lags E(t-L), future leads E(t+1..t+2), and baseline design D."""
    common_start = max(conv_daily.index.min(), evc_daily.index.min())
    common_end = min(conv_daily.index.max(), evc_daily.index.max())
    if common_start > common_end:
        raise ValueError("No overlapping dates between conversions and EVC series.")

    full_dates = pd.date_range(common_start, common_end, freq="D")
    conv_al = conv_daily.reindex(full_dates, fill_value=0.0).astype(float)
    evc_al = evc_daily.reindex(full_dates, fill_value=0.0).astype(float)

    # Mandatory direction: E_L(t) = E(t - L) -> pandas shift(+L)
    lag_df = pd.DataFrame({f"lag_{L}": evc_al.shift(L) for L in range(max_lag + 1)}, index=full_dates)
    lead_s = 0.5 * (evc_al.shift(-1) + evc_al.shift(-2))

    # Drop first max_lag days (for lags) and last 2 days (for leads)
    valid_mask = lag_df.notna().all(axis=1) & lead_s.notna()
    dates = full_dates[valid_mask]
    if len(dates) < 10:
        raise ValueError(f"Analysis window too short after lag/lead trimming ({len(dates)} days).")

    T = len(dates)
    intercept = np.ones((T, 1), dtype=float)
    trend = (np.arange(T, dtype=float) / max(T, 1)).reshape(-1, 1)
    dows = dates.dayofweek.to_numpy()
    dow_dummies = np.column_stack([(dows == d).astype(float) for d in range(1, 7)])

    d_parts = [intercept, trend, dow_dummies]
    if controls is not None:
        missing_dates = dates.difference(controls.index)
        if len(missing_dates) > 0:
            raise ValueError(
                f"Controls file is missing {len(missing_dates)} dates in the analysis window "
                f"(first missing: {missing_dates[0].date()})."
            )
        ctrl_vals = controls.loc[dates].to_numpy(dtype=float)
        if np.isnan(ctrl_vals).any():
            raise ValueError("Controls contain NaN values inside the analysis window.")
        d_parts.append(ctrl_vals)

    D = np.hstack(d_parts)
    return PreparedDesign(
        dates=dates,
        channels=list(conv_al.columns),
        Y=conv_al.loc[dates].to_numpy(dtype=float),
        E=evc_al.loc[dates].to_numpy(dtype=float),
        E_lags=lag_df.loc[dates].to_numpy(dtype=float),
        z_lead=lead_s.loc[dates].to_numpy(dtype=float),
        D=D,
    )


def _project_simplex_eq(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection onto the probability simplex {k >= 0, sum(k) = 1}."""
    u = np.asarray(v, dtype=float)
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s) - 1.0
    ind = np.arange(1, len(u) + 1, dtype=float)
    cond = s - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(u, 1.0 / len(u))
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(u - theta, 0.0)


def _solve_k_simplex(E_tilde: np.ndarray, Y_tilde: np.ndarray, w: np.ndarray, k_init: np.ndarray) -> np.ndarray:
    """Minimize sum_c ||y_tilde_c - w_c * (E_tilde @ k)||^2 over simplex {k >= 0, sum(k) = 1}."""
    sw = float(w @ w)
    if sw <= 1e-12:
        return k_init
    y_bar = (Y_tilde @ w) / sw
    G = E_tilde.T @ E_tilde
    g = E_tilde.T @ y_bar
    scale = float(np.max(np.abs(G)))
    if scale <= 1e-12:
        return k_init
    Gn = G / scale
    gn = g / scale
    L_len = len(k_init)

    def obj(kk: np.ndarray) -> float:
        return float(0.5 * kk @ Gn @ kk - gn @ kk)

    def jac(kk: np.ndarray) -> np.ndarray:
        return Gn @ kk - gn

    res = minimize(
        obj,
        x0=k_init,
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * L_len,
        constraints=[{"type": "eq", "fun": lambda kk: float(np.sum(kk) - 1.0), "jac": lambda kk: np.ones(L_len)}],
        options={"ftol": 1e-10, "maxiter": 60, "disp": False},
    )
    if res.success and np.all(np.isfinite(res.x)):
        k_new = np.clip(res.x, 0.0, 1.0)
        s = float(k_new.sum())
        return k_new / s if s > 0 else np.full(L_len, 1.0 / L_len)

    # Fallback: projected gradient on the normalized simplex QP if SLSQP line-search stalls
    L_lip = max(float(np.linalg.norm(Gn, 2)), 1e-12)
    step = 1.0 / L_lip
    kk = _project_simplex_eq(res.x if np.all(np.isfinite(res.x)) else k_init)
    if obj(k_init) < obj(kk):
        kk = k_init.copy()
    for _ in range(200):
        k_next = _project_simplex_eq(kk - step * jac(kk))
        if float(np.max(np.abs(k_next - kk))) < 1e-10:
            kk = k_next
            break
        kk = k_next
    return kk


def _fit_core(
    Y: np.ndarray,
    E_lags: np.ndarray,
    D: np.ndarray,
    z_lead: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Single fit on (Y, E_lags, D, z_lead). Returns (w, w_unconstrained, k, lead_coef, converged)."""
    C = Y.shape[1]
    L_len = E_lags.shape[1]

    # Frisch-Waugh-Lovell residualization in one lstsq call
    M = np.hstack([Y, E_lags, z_lead.reshape(-1, 1)])
    coef_D, _, _, _ = np.linalg.lstsq(D, M, rcond=None)
    M_tilde = M - D @ coef_D

    Y_tilde = M_tilde[:, :C]
    E_tilde = M_tilde[:, C : C + L_len]
    z_tilde = M_tilde[:, -1]

    k = np.full(L_len, 1.0 / L_len, dtype=float)
    w = np.zeros(C, dtype=float)
    w_unc = np.zeros(C, dtype=float)
    converged = False

    for _ in range(max_iter):
        x_tilde = E_tilde @ k
        denom = float(x_tilde @ x_tilde)
        if denom <= 1e-12:
            w_new = np.zeros(C, dtype=float)
            w_unc = np.zeros(C, dtype=float)
            converged = True
            w = w_new
            break

        w_unc = (Y_tilde.T @ x_tilde) / denom
        w_new = project_simplex_ball(w_unc)

        if L_len > 1 and float(w_new.sum()) > 1e-12:
            k_new = _solve_k_simplex(E_tilde, Y_tilde, w_new, k)
        else:
            k_new = k

        diff = max(float(np.max(np.abs(w_new - w))), float(np.max(np.abs(k_new - k))))
        w, k = w_new, k_new
        if diff < tol:
            converged = True
            break

    # Lead falsification OLS: Y_tilde ~ [x_tilde, z_tilde]
    x_final = E_tilde @ k
    xz = np.column_stack([x_final, z_tilde])
    xz_coef, _, _, _ = np.linalg.lstsq(xz, Y_tilde, rcond=None)
    lead_coef = xz_coef[1, :]

    if float(w.sum()) <= 1e-12:
        k_out = np.full(L_len, np.nan, dtype=float)
    else:
        k_out = k

    return w, w_unc, k_out, lead_coef, converged


def assign_confidence_label(
    ci_low: float,
    lead_flag: bool,
    powered: bool | None,
) -> str:
    """Deterministic confidence label per spec §7.7."""
    # Only flag Confounded when the channel actually shows a positive lower-bound share.
    if lead_flag and ci_low > 0.01:
        return "Confounded"
    if ci_low <= 0.01 or powered is False:
        return "Inconclusive"
    if ci_low >= 0.10:
        return "Strong"
    return "Directional"


def fit_model(
    conv_daily: pd.DataFrame,
    evc_daily: pd.Series,
    config: Config | None = None,
    controls: pd.DataFrame | None = None,
    bootstrap_override: int | None = None,
    seed_override: int | None = None,
) -> FitOutput:
    """Fit constrained EVC landing model with moving-block bootstrap and lead falsification."""
    cfg = config or Config()
    n_boot = cfg.bootstrap if bootstrap_override is None else bootstrap_override
    seed = cfg.seed if seed_override is None else seed_override

    design = build_design(conv_daily, evc_daily, max_lag=cfg.max_lag, controls=controls)
    T, C = design.Y.shape
    L_len = design.E_lags.shape[1]

    w, w_unc, k_out, lead_coef, converged = _fit_core(design.Y, design.E_lags, design.D, design.z_lead)
    warnings: list[str] = []
    if float(w.sum()) <= 1e-12:
        warnings.append("No EVC signal detected in any channel.")

    # Reconstruct baseline & EVC component + partial R^2
    k_eff = np.full(L_len, 1.0 / L_len) if np.isnan(k_out).any() else k_out
    x_full = design.E_lags @ k_eff
    evc_comp = np.outer(x_full, w)
    beta_D, _, _, _ = np.linalg.lstsq(design.D, design.Y - evc_comp, rcond=None)
    baseline_fit = design.D @ beta_D

    # Partial R^2 of EVC term on residualized Y
    M = np.hstack([design.Y, x_full.reshape(-1, 1)])
    coef_D_m, _, _, _ = np.linalg.lstsq(design.D, M, rcond=None)
    M_t = M - design.D @ coef_D_m
    Y_t, x_t = M_t[:, :C], M_t[:, -1]
    partial_r2 = np.zeros(C, dtype=float)
    for c in range(C):
        sst = float(Y_t[:, c] @ Y_t[:, c])
        if sst > 1e-12:
            resid = Y_t[:, c] - w[c] * x_t
            partial_r2[c] = max(0.0, 1.0 - float(resid @ resid) / sst)

    # Moving-block bootstrap
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        blen = min(cfg.block_len, max(1, T // 2))
        n_blocks = int(math.ceil(T / blen))
        max_start = max(1, T - blen + 1)

        W_boot = np.empty((n_boot, C), dtype=float)
        NV_boot = np.empty(n_boot, dtype=float)
        K_boot = np.full((n_boot, L_len), np.nan, dtype=float)
        Lead_boot = np.empty((n_boot, C), dtype=float)

        offsets = np.arange(blen)
        for b in range(n_boot):
            starts = rng.integers(0, max_start, size=n_blocks)
            idx = (starts[:, None] + offsets[None, :]).ravel()[:T]
            wb, _, kb, lb, _ = _fit_core(design.Y[idx], design.E_lags[idx], design.D[idx], design.z_lead[idx])
            W_boot[b] = wb
            NV_boot[b] = max(0.0, 1.0 - float(wb.sum()))
            K_boot[b] = kb
            Lead_boot[b] = lb

        ci_low = np.percentile(W_boot, 5, axis=0)
        ci_high = np.percentile(W_boot, 95, axis=0)
        nv_ci = (float(np.percentile(NV_boot, 5)), float(np.percentile(NV_boot, 95)))

        max_w = W_boot.max(axis=1)
        argmax_w = W_boot.argmax(axis=1)
        p_top = np.array([float(np.mean((argmax_w == c) & (max_w > 0))) for c in range(C)], dtype=float)
        p_material = np.mean(W_boot >= 0.05, axis=0).astype(float)

        lead_ci_low = np.percentile(Lead_boot, 5, axis=0)
        lead_ci_high = np.percentile(Lead_boot, 95, axis=0)

        valid_k = ~np.isnan(K_boot).any(axis=1)
        if np.any(valid_k):
            k_ci_low = np.percentile(K_boot[valid_k], 5, axis=0)
            k_ci_high = np.percentile(K_boot[valid_k], 95, axis=0)
        else:
            k_ci_low = np.full(L_len, np.nan)
            k_ci_high = np.full(L_len, np.nan)
    else:
        ci_low = w.copy()
        ci_high = w.copy()
        nv_ci = (max(0.0, 1.0 - float(w.sum())), max(0.0, 1.0 - float(w.sum())))
        p_top = np.zeros(C, dtype=float)
        if float(w.sum()) > 0:
            p_top[int(np.argmax(w))] = 1.0
        p_material = (w >= 0.05).astype(float)
        lead_ci_low = lead_coef.copy()
        lead_ci_high = lead_coef.copy()
        k_ci_low = k_out.copy()
        k_ci_high = k_out.copy()

    lead_flag = (lead_ci_low > 0.0) & (lead_coef > 0.0)
    confounded = bool(np.any(lead_flag & (w >= 0.10)))

    return FitOutput(
        dates=design.dates,
        channels=design.channels,
        shares=w,
        unconstrained_shares=w_unc,
        ci_low=ci_low,
        ci_high=ci_high,
        p_top=p_top,
        p_material=p_material,
        partial_r2=partial_r2,
        lead_coef=lead_coef,
        lead_ci_low=lead_ci_low,
        lead_ci_high=lead_ci_high,
        lead_flag=lead_flag,
        not_visible=max(0.0, 1.0 - float(w.sum())),
        not_visible_ci=nv_ci,
        k=k_out,
        k_ci_low=k_ci_low,
        k_ci_high=k_ci_high,
        confounded=confounded,
        converged=converged,
        baseline_fit=baseline_fit,
        evc_component=evc_comp,
        evc_series=design.E,
        actual_y=design.Y,
        warnings=warnings,
    )
