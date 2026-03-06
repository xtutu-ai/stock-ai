# select.py
import os
import time
import re
import math
import requests
import pandas as pd
import numpy as np

# chart
import matplotlib.pyplot as plt

BASE_URL = "http://127.0.0.1:8000"

# =========================
# Parameters
# =========================
MIN_PRICE = 5.0

MIN_AMOUNT_MAIN = 5e8
TOPN_MAIN = 700

MIN_AMOUNT_SECOND = 1.5e8
TOPN_SECOND_POOL = 1600

TAIL = 280
LOOKBACK = 120
SLEEP = 0.03
TOPK_OUT = 30

ATR_SOFT_MAIN = 0.10
ATR_SOFT_GROWTH = 0.14
ATR_HARD_MAX = 0.18

RET_120_SOFT = 1.2
RET_120_HARD = 2.2

VCP_MAX_DIST20 = 0.10
VCP_MIN_DIST20 = 0.003
VCP_MAX_TIGHT10 = 0.08
VCP_ATR_SHRINK_RATIO = 0.96
VCP_MIN_RET120 = 0.20

# fused weights
FUSE_W_SCORE = 0.72
FUSE_W_RANK  = 0.28

# pro filters
CHASE_STRICT_START_PCT_MAIN = 7.0     # main board start penalize
CHASE_STRICT_START_PCT_GEM  = 10.0    # 创业板 start penalize
BIAS20_HARD = 0.15                    # (close-ma20)/ma20 > 15% => penalize
SMART_MONEY_MIN = -0.35               # smart_money < -0.35 => filter out
RSRS_MIN_R2 = 0.55                    # too low r2 => lower confidence

# momentum / structure enhancements
PATH_R2_GOOD = 0.65                 # path linearity considered good

# =========================
# HTTP helpers
# =========================
def http_get(path, params=None, timeout=60):
    return requests.get(BASE_URL + path, params=params, timeout=timeout)

def http_post(path, params=None, timeout=240):
    return requests.post(BASE_URL + path, params=params, timeout=timeout)

def load_universe_df() -> pd.DataFrame:
    """
    Prefer reading local universe parquet path from /health.cache_dir.
    """
    r = http_get("/health", timeout=10)
    r.raise_for_status()
    cache_dir = r.json()["cache_dir"]
    path = f"{cache_dir}/universe_latest.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"universe parquet not found: {path}. "
            f"Call /refresh_universe?trade_date=YYYYMMDD first."
        )
    return pd.read_parquet(path)

def ensure_hist_cached(code: str, start_date="20180101"):
    r = http_get(f"/hist/{code}", params={"tail": 5}, timeout=30)
    if r.status_code == 404:
        rr = http_post(f"/daily_refresh/{code}", params={"start_date": start_date}, timeout=240)
        if not rr.ok:
            try:
                print(f"[daily_refresh failed] {code}: {rr.status_code} {rr.json()}")
            except Exception:
                print(f"[daily_refresh failed] {code}: {rr.status_code} {rr.text}")
            rr.raise_for_status()
    elif not r.ok:
        r.raise_for_status()

def get_hist_df(code: str, tail: int = TAIL) -> pd.DataFrame:
    r = http_get(f"/hist/{code}", params={"tail": tail}, timeout=60)
    if r.status_code == 404:
        ensure_hist_cached(code)
        r = http_get(f"/hist/{code}", params={"tail": tail}, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["tail"])

# =========================
# Utility
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def is_chuangye(code: str) -> bool:
    return str(code).startswith(("300", "301"))

def board_limit_abs_pct(code: str) -> float:
    return 20.0 if is_chuangye(code) else 10.0

def atr_soft_limit(code: str) -> float:
    return ATR_SOFT_GROWTH if is_chuangye(code) else ATR_SOFT_MAIN

def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("（", "(").replace("）", ")").replace(" ", "")
    return s

def pick_ohlc_columns(h: pd.DataFrame):
    """
    ✅ robust: support both EN and CN columns
    Return: (open_col_or_None, close_col, high_col, low_col) or None
    """
    cols = set(h.columns)

    # english
    if {"open", "close", "high", "low"}.issubset(cols):
        return "open", "close", "high", "low"
    if {"close", "high", "low"}.issubset(cols):
        return None, "close", "high", "low"

    # chinese
    if {"开盘", "收盘", "最高", "最低"}.issubset(cols):
        return "开盘", "收盘", "最高", "最低"
    if {"收盘", "最高", "最低"}.issubset(cols):
        return None, "收盘", "最高", "最低"

    # alternative chinese
    if {"开盘价", "收盘价", "最高价", "最低价"}.issubset(cols):
        return "开盘价", "收盘价", "最高价", "最低价"
    if {"收盘价", "最高价", "最低价"}.issubset(cols):
        return None, "收盘价", "最高价", "最低价"

    return None

def pick_volume_or_amount_columns_auto(h: pd.DataFrame):
    cols = list(h.columns)
    if not cols:
        return None, None
    ncols = [_norm(c) for c in cols]

    def find_best(patterns):
        for pat in patterns:
            rx = re.compile(pat)
            for i, nc in enumerate(ncols):
                if rx.search(nc):
                    return cols[i]
        return None

    amount_col = find_best([r"amount_yuan", r"\bamount\b", r"成交额", r"成交金额"])
    vol_col = find_best([r"\bvol\b", r"volume", r"成交量"])
    return amount_col, vol_col

def calc_vol_ratio_from_hist(h: pd.DataFrame) -> float:
    amount_col, vol_col = pick_volume_or_amount_columns_auto(h)
    series = None
    if amount_col is not None:
        series = pd.to_numeric(h[amount_col], errors="coerce")
    elif vol_col is not None:
        series = pd.to_numeric(h[vol_col], errors="coerce")

    if series is None or len(series.dropna()) < 25:
        return np.nan

    ma20 = series.rolling(20).mean().iloc[-1]
    last = series.iloc[-1]
    if pd.isna(ma20) or float(ma20) == 0 or pd.isna(last):
        return np.nan
    return float(last / (float(ma20) + 1e-9))

# =========================
# Indicators
# =========================
def calc_atr_series_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n=14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def close_tightness_10(close: pd.Series) -> float:
    c = pd.to_numeric(close.tail(10), errors="coerce").dropna()
    if len(c) < 8:
        return np.nan
    return float((c.max() - c.min()) / c.iloc[-1])

def soft_trend_score(ret_120: float) -> float:
    if not np.isfinite(ret_120) or ret_120 <= 0:
        return 0.0
    base = clamp(ret_120 / 1.2, 0, 1.2)
    if ret_120 > RET_120_SOFT:
        base -= clamp((ret_120 - RET_120_SOFT) / 0.9, 0, 1) * 0.25
    if ret_120 > RET_120_HARD:
        base -= clamp((ret_120 - RET_120_HARD) / 1.2, 0, 1) * 0.20
    return clamp(base, 0, 1.2)

def vol_ratio_score_expand(vr: float) -> float:
    if not np.isfinite(vr):
        return 0.35
    if vr < 0.7:
        return 0.25
    if vr < 1.0:
        return 0.45
    if vr < 1.3:
        return 0.70
    if vr < 1.8:
        return 1.00
    return 0.88

def vol_ratio_score_contract(vr: float) -> float:
    if not np.isfinite(vr):
        return 0.40
    if vr <= 0.45:
        return 1.00
    if vr <= 0.60:
        return 0.90
    if vr <= 0.75:
        return 0.80
    if vr <= 0.90:
        return 0.60
    return 0.40

def dist_to_high_score(dist: float, vr: float, mode_expand=True) -> float:
    if not np.isfinite(dist):
        return 0.45
    if mode_expand:
        if dist < 0.01:
            return 1.00 if (np.isfinite(vr) and vr >= 1.3) else 0.35
        if dist <= 0.04:
            return 0.95
        if dist <= 0.10:
            return 0.75
        if dist <= 0.16:
            return 0.55
        return 0.35
    if dist < 0.003:
        return 0.50
    if dist <= 0.08:
        return 1.00
    if dist <= 0.12:
        return 0.75
    if dist <= 0.16:
        return 0.55
    return 0.35

def breakout_gate_abs_pct(code: str, dist20: float) -> float:
    limit = board_limit_abs_pct(code)
    normal = min(limit * 0.60, limit - 0.5)
    breakout = min(limit * 0.95, limit - 0.5)
    if np.isfinite(dist20) and dist20 <= 0.03:
        return breakout
    return normal

def atr_penalty(atr_ratio: float, code: str) -> float:
    if not np.isfinite(atr_ratio):
        return 0.0
    if atr_ratio >= ATR_HARD_MAX:
        return 0.0
    soft = atr_soft_limit(code)
    if atr_ratio <= soft:
        return 1.0
    t = clamp((atr_ratio - soft) / (ATR_HARD_MAX - soft), 0, 1)
    return float(1.0 - 0.4 * t)

def blowoff_penalty(vr: float, pct: float, dist20: float, ret_120: float) -> float:
    if not (np.isfinite(vr) and np.isfinite(pct) and np.isfinite(dist20) and np.isfinite(ret_120)):
        return 0.0
    if vr > 2.2 and abs(pct) > 8.0 and dist20 < 0.015 and ret_120 > 1.2:
        return 0.12
    if vr > 2.8 and abs(pct) > 9.0 and dist20 < 0.010:
        return 0.18
    return 0.0

def chase_penalty(pct: float, code: str) -> float:
    """
    Penalize high same-day move (T+1 risk).
    Return 0~0.24
    """
    if not np.isfinite(pct):
        return 0.0
    limit = board_limit_abs_pct(code)
    start = CHASE_STRICT_START_PCT_GEM if is_chuangye(code) else CHASE_STRICT_START_PCT_MAIN
    start = min(start, 0.75 * limit)
    very_hot = 0.92 * limit

    p = abs(float(pct))
    if p < start:
        return 0.0
    if p >= very_hot:
        return 0.24

    t = clamp((p - start) / (very_hot - start), 0, 1)
    return float(0.08 + 0.12 * t)

def bias20_penalty(last: float, ma20: float) -> float:
    """
    Overheat penalty when price too far above MA20.
    """
    if not (np.isfinite(last) and np.isfinite(ma20) and ma20 > 0):
        return 0.0
    bias = (last - ma20) / ma20
    if bias <= 0:
        return 0.0
    if bias >= BIAS20_HARD:
        return 0.18
    t = clamp(bias / BIAS20_HARD, 0, 1)
    return float(0.04 + 0.10 * t)

def smart_money_factor(open_: float, close: float, high: float, low: float) -> float:
    """
    (close-open)/(high-low) in [-1,1], higher means stronger close.
    """
    if not (np.isfinite(open_) and np.isfinite(close) and np.isfinite(high) and np.isfinite(low)):
        return np.nan
    rng = (high - low)
    if rng <= 0:
        return np.nan
    return float((close - open_) / (rng + 1e-9))

def vcp_score(vr, dist20, last, ma60, tight10, atr_now, atr_ago) -> float:
    if not (np.isfinite(dist20) and np.isfinite(last) and np.isfinite(ma60) and np.isfinite(tight10)):
        return 0.0
    if last < ma60:
        return 0.0
    if dist20 < VCP_MIN_DIST20 or dist20 > VCP_MAX_DIST20:
        return 0.0
    if tight10 > VCP_MAX_TIGHT10:
        return 0.0
    if (not np.isfinite(atr_now)) or (not np.isfinite(atr_ago)) or atr_ago <= 0:
        return 0.0
    if atr_now > atr_ago * VCP_ATR_SHRINK_RATIO:
        return 0.0

    contract = vol_ratio_score_contract(vr)
    dists = dist_to_high_score(dist20, vr, mode_expand=False)
    tight_bonus = 1.0 - clamp(tight10 / VCP_MAX_TIGHT10, 0, 1) * 0.4
    score = 0.50 * contract + 0.30 * dists + 0.20 * tight_bonus
    return float(clamp(score, 0, 1.2))

# =========================
# Momentum path quality
# =========================
def momentum_path_quality(close: pd.Series, lookback=120):
    """
    Use log-price path to avoid over-rewarding nominal price level.
    Return:
      r2: linear fit quality of log(close) vs time
      max_dd: max drawdown over the window
    """
    c = pd.to_numeric(close.tail(lookback), errors="coerce").dropna()
    if len(c) < max(30, lookback // 3):
        return np.nan, np.nan

    y = np.log(c.clip(lower=1e-9).values.astype(float))
    x = np.arange(len(y), dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0:
        return np.nan, np.nan

    beta = np.sum((x - x_mean) * (y - y_mean)) / denom
    alpha = y_mean - beta * x_mean
    y_hat = beta * x + alpha
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    px = c.values.astype(float)
    peak = np.maximum.accumulate(px)
    dd = (peak - px) / (peak + 1e-9)
    max_dd = float(np.nanmax(dd)) if len(dd) else np.nan
    return r2, max_dd


def path_quality_score(r2: float, max_dd: float) -> float:
    """Blend path smoothness and drawdown into [0, 1]."""
    if not (np.isfinite(r2) and np.isfinite(max_dd)):
        return 0.50

    r2_s = clamp(r2 / max(PATH_R2_GOOD, 1e-9), 0, 1.0)

    if max_dd <= 0.12:
        dd_s = 1.0
    elif max_dd <= 0.22:
        dd_s = 1.0 - (max_dd - 0.12) / 0.10 * 0.25
    elif max_dd <= 0.35:
        dd_s = 0.75 - (max_dd - 0.22) / 0.13 * 0.35
    else:
        dd_s = 0.40 - clamp((max_dd - 0.35) / 0.20, 0, 1) * 0.25

    return float(clamp(0.55 * r2_s + 0.45 * dd_s, 0, 1.0))


# =========================
# MA dynamics (continuous features, keep binary structure too)
# =========================
def ma_dynamics(close: pd.Series) -> dict:
    c = pd.to_numeric(close, errors="coerce")
    out = {"ma20_slope": np.nan, "ma_convergence": np.nan, "ma20_above_days": 0}
    if len(c) < 65:
        return out

    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()

    if pd.notna(ma20.iloc[-1]) and pd.notna(ma20.iloc[-6]) and float(ma20.iloc[-6]) > 0:
        slope_5d = (float(ma20.iloc[-1]) - float(ma20.iloc[-6])) / float(ma20.iloc[-6])
        out["ma20_slope"] = float(slope_5d * (252.0 / 5.0))

    last = float(c.iloc[-1]) if len(c) else np.nan
    if pd.notna(ma20.iloc[-1]) and pd.notna(ma60.iloc[-1]) and np.isfinite(last) and last > 0:
        out["ma_convergence"] = abs(float(ma20.iloc[-1]) - float(ma60.iloc[-1])) / last

    tail20 = c.tail(20)
    ma20_t = ma20.tail(20)
    out["ma20_above_days"] = int((tail20 > ma20_t).sum())
    return out


def ma_slope_score(slope: float) -> float:
    if not np.isfinite(slope):
        return 0.50
    if slope <= 0:
        return float(clamp(0.28 + slope * 0.35, 0.05, 0.35))
    return float(clamp(0.52 + slope * 0.40, 0.52, 1.00))


def ma_converge_score(convergence: float, ma20_ge_ma60: bool) -> float:
    if not np.isfinite(convergence):
        return 0.50
    if convergence < 0.02:
        return 0.92 if ma20_ge_ma60 else 0.68
    if convergence < 0.05:
        return 0.74 if ma20_ge_ma60 else 0.52
    return 0.56 if ma20_ge_ma60 else 0.36


# =========================
# Interaction bonus (small, diagnostic-friendly)
# =========================
def interaction_bonus(dist20: float, vr: float, ma_convergence: float,
                      ma20_ge_ma60: bool, tight10: float, ma20_slope: float) -> dict:
    bonus = {"breakout_synergy": 0.0, "vcp_synergy": 0.0, "trend_confirm": 0.0}

    if np.isfinite(dist20) and dist20 < 0.03 and np.isfinite(vr) and vr >= 1.3:
        bonus["breakout_synergy"] = float(clamp(0.03 + 0.04 * (vr - 1.3), 0, 0.09))

    if (np.isfinite(ma_convergence) and ma_convergence < 0.03 and
            np.isfinite(tight10) and tight10 < 0.06 and np.isfinite(vr) and vr <= 0.90):
        bonus["vcp_synergy"] = float(clamp(0.03 + (0.90 - vr) * 0.06, 0, 0.08))

    if ma20_ge_ma60 and np.isfinite(ma20_slope) and ma20_slope > 0:
        bonus["trend_confirm"] = float(clamp(0.02 + ma20_slope * 0.02, 0, 0.07))

    return bonus

# =========================
# RSRS (no scipy)
# =========================
def calc_rsrs_score(df: pd.DataFrame, N=18, M=180):
    """
    RSRS: rolling regression High ~ beta * Low + alpha
    score = zscore(beta) * r2
    """
    d = df.copy()
    if not {"high", "low"}.issubset(set(d.columns)):
        return np.nan, np.nan

    high = pd.to_numeric(d["high"], errors="coerce").values
    low = pd.to_numeric(d["low"], errors="coerce").values

    betas = np.full(len(d), np.nan, dtype=float)
    r2s = np.full(len(d), np.nan, dtype=float)

    for i in range(N, len(d) + 1):
        y = high[i - N:i]
        x = low[i - N:i]
        if np.isnan(y).any() or np.isnan(x).any():
            continue

        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom <= 0:
            continue
        beta = np.sum((x - x_mean) * (y - y_mean)) / denom
        alpha = y_mean - beta * x_mean

        y_hat = beta * x + alpha
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot <= 0:
            continue

        r2 = 1.0 - ss_res / ss_tot
        betas[i - 1] = beta
        r2s[i - 1] = r2

    s_beta = pd.Series(betas)
    mean_beta = s_beta.rolling(window=M, min_periods=max(20, N + 2)).mean()
    std_beta = s_beta.rolling(window=M, min_periods=max(20, N + 2)).std()

    z = (s_beta - mean_beta) / std_beta
    rsrs_score = float((z.iloc[-1] * r2s[-1]) if np.isfinite(z.iloc[-1]) and np.isfinite(r2s[-1]) else np.nan)
    rsrs_r2 = float(r2s[-1]) if np.isfinite(r2s[-1]) else np.nan
    return rsrs_score, rsrs_r2

# =========================
# Market regime (simple & accurate with your available data)
# =========================
def detect_market_regime(universe_df: pd.DataFrame) -> str:
    """
    Use only universe snapshot:
    - breadth: pct>0 ratio
    - liquidity: median amount_yuan
    - return: median pct
    Return: 'bull' / 'sideways' / 'bear'
    """
    d = universe_df.copy()
    pct = pd.to_numeric(d["pct"], errors="coerce")
    amt = pd.to_numeric(d["amount_yuan"], errors="coerce")

    valid = pct.notna() & amt.notna()
    if valid.sum() < 300:
        return "sideways"

    breadth = float((pct[valid] > 0).mean())
    med_pct = float(pct[valid].median())
    med_amt = float(amt[valid].median())

    if breadth >= 0.58 and med_pct > 0.35 and med_amt >= 1.2e8:
        return "bull"
    if breadth <= 0.42 and med_pct < -0.35:
        return "bear"
    return "sideways"

def apply_regime_weights(regime: str):
    """
    Tiny weight tweaks (do not change core model).
    """
    if regime == "bear":
        return {"breakout_mul": 0.86, "cont_mul": 0.86, "vcp_mul": 1.05, "early_mul": 0.95}
    if regime == "bull":
        return {"breakout_mul": 1.05, "cont_mul": 1.05, "vcp_mul": 0.95, "early_mul": 1.00}
    return {"breakout_mul": 1.00, "cont_mul": 1.00, "vcp_mul": 1.00, "early_mul": 1.00}

# =========================
# Rank scoring (robust)
# =========================
def build_rank_score(df: pd.DataFrame) -> pd.Series:
    d = df.copy()

    need_cols = ["ret_120", "atr_ratio", "dist20", "tight10", "vol_ratio", "amount_yuan", "rsrs_score", "path_quality", "pe_ttm"]
    for c in need_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    def _safe_winsor(s: pd.Series, p=0.01):
        x = pd.to_numeric(s, errors="coerce")
        if x.notna().sum() < 10:
            return x
        lo = x.quantile(p)
        hi = x.quantile(1 - p)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return x
        return x.clip(lo, hi)

    for c in ["ret_120", "atr_ratio", "dist20", "tight10", "vol_ratio", "amount_yuan", "rsrs_score", "path_quality"]:
        if c in d.columns:
            d[c] = _safe_winsor(d[c], 0.01)

    has_pe = "pe_ttm" in d.columns
    if has_pe:
        d.loc[d["pe_ttm"] <= 0, "pe_ttm"] = np.nan

    out = pd.Series(np.nan, index=d.index, dtype=float)

    def r01(s: pd.Series, higher_better=True) -> pd.Series:
        r = pd.to_numeric(s, errors="coerce").rank(pct=True, ascending=not higher_better)
        return r.fillna(0.5)

    for model in d["model"].dropna().unique():
        sub = d[d["model"] == model].copy()
        if sub.empty:
            continue

        r_mom  = r01(sub["ret_120"], True)
        r_atr  = r01(sub["atr_ratio"], False)
        r_liq  = r01(np.log10(sub["amount_yuan"].fillna(0) + 1), True)
        r_rsrs = r01(sub["rsrs_score"], True)
        r_pq   = r01(sub.get("path_quality", 0.5), True)

        if model == "VCP_缩量收敛前夜":
            r_tight = r01(sub["tight10"], False)
            r_dist  = r01(sub["dist20"], True)
            r_vr_c  = r01(sub["vol_ratio"], False)
            score = (
                0.18 * r_mom + 0.14 * r_atr + 0.10 * r_liq + 0.16 * r_rsrs
                + 0.20 * r_tight + 0.08 * r_dist + 0.08 * r_vr_c + 0.06 * r_pq
            )
        elif model == "BREAKOUT_即将启动":
            r_dist  = r01(sub["dist20"], False)
            r_vr_e  = r01(sub["vol_ratio"], True)
            score = 0.20*r_mom + 0.14*r_atr + 0.12*r_liq + 0.20*r_rsrs + 0.18*r_dist + 0.10*r_vr_e + 0.06*r_pq
        elif model == "EARLY_TREND_主线前期":
            r_dist  = r01(sub["dist20"], False)
            score = 0.26*r_mom + 0.16*r_atr + 0.16*r_liq + 0.20*r_rsrs + 0.14*r_dist + 0.08*r_pq
        else:
            r_dist  = r01(sub["dist20"], False)
            r_vr_e  = r01(sub["vol_ratio"], True)
            score = 0.22*r_mom + 0.14*r_atr + 0.12*r_liq + 0.20*r_rsrs + 0.16*r_dist + 0.10*r_vr_e + 0.06*r_pq

        if has_pe:
            r_value = r01(1.0 / sub["pe_ttm"], True)
            score = 0.88 * score + 0.12 * r_value

        out.loc[sub.index] = score.astype(float)

    out = out.fillna(0.5)
    return (out * 1.25).clip(0, 1.25)

# =========================
# Charts
# =========================
def save_charts(res: pd.DataFrame, out_dir: str = "./out_charts"):
    os.makedirs(out_dir, exist_ok=True)
    ts_tag = time.strftime("%Y%m%d_%H%M%S")

    top = res.head(20).copy()
    top = top.iloc[::-1]  # for barh

    # 1) barh of score_fused top20
    plt.figure()
    plt.barh(top["code"] + " " + top["name"], top["score_fused"])
    plt.xlabel("score_fused")
    plt.title("Top20 score_fused")
    p1 = os.path.join(out_dir, f"top20_score_fused_{ts_tag}.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()

    # 2) scatter score_fused vs pct (chase risk)
    plt.figure()
    x = pd.to_numeric(res["pct"], errors="coerce")
    y = pd.to_numeric(res["score_fused"], errors="coerce")
    plt.scatter(x, y, s=12)
    plt.xlabel("pct (today)")
    plt.ylabel("score_fused")
    plt.title("score_fused vs pct (chase-risk view)")
    p2 = os.path.join(out_dir, f"scatter_score_vs_pct_{ts_tag}.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()

    print(f"\n[Charts] saved:")
    print(f" - {os.path.abspath(p1)}")
    print(f" - {os.path.abspath(p2)}")

# =========================
# Main
# =========================
def run_selector(topk_out=TOPK_OUT, make_charts=False):
    # 1) load universe
    spot = load_universe_df()

    # 2) verify trade_date freshness via /health meta (accurate!)
    try:
        h = http_get("/health", timeout=10).json()
        meta_td = (h.get("universe_meta") or {}).get("trade_date")
        if meta_td:
            print(f"[Universe] service meta trade_date = {meta_td}")
        if "trade_date" in spot.columns and spot["trade_date"].astype(str).nunique() == 1:
            print(f"[Universe] parquet trade_date = {spot['trade_date'].astype(str).iloc[0]}")
    except Exception:
        pass

    df = spot.copy()
    # ensure expected schema
    for col in ["code", "name", "price", "pct", "amount_yuan"]:
        if col not in df.columns:
            raise RuntimeError(f"universe_latest.parquet missing required column: {col}")

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df["amount_yuan"] = pd.to_numeric(df["amount_yuan"], errors="coerce")

    for c in ["pe_ttm", "pb", "turnover_rate", "volume_ratio_basic"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    print("\n[Sanity] Top amount_yuan:")
    print(df.sort_values("amount_yuan", ascending=False).head(5)[["code","name","amount_yuan","price","pct"]].to_string(index=False))

    # 3) market regime
    regime = detect_market_regime(df)
    w = apply_regime_weights(regime)
    print(f"\n[Regime] {regime} weights={w}")

    base = df[(df["price"] >= MIN_PRICE) & (~df["name"].str.contains("ST", na=False))].copy()
    if base.empty:
        print("初筛为空：检查 MIN_PRICE 或 universe 数据。")
        return

    # optional: filter 科创板 688
    base = base[~base["code"].str.startswith("68")].copy()

    base = base.sort_values("amount_yuan", ascending=False).head(TOPN_SECOND_POOL).copy()
    base["is_main_universe"] = False
    main_n = min(TOPN_MAIN, len(base))
    base.loc[base.iloc[:main_n].index, "is_main_universe"] = True

    buckets = {"VCP_缩量收敛前夜": [], "BREAKOUT_即将启动": [], "EARLY_TREND_主线前期": [], "CONTINUATION_主升延续": []}
    all_candidates = []
    total = len(base)
    fail = 0

    for i, r in enumerate(base.itertuples(index=False), start=1):
        code = r.code
        name = r.name
        price_now = float(r.price) if np.isfinite(r.price) else np.nan
        pct_now = float(r.pct) if np.isfinite(r.pct) else np.nan
        amt_now = float(r.amount_yuan) if np.isfinite(r.amount_yuan) else np.nan
        is_main = bool(r.is_main_universe)

        pe_ttm = getattr(r, "pe_ttm", np.nan)
        pb = getattr(r, "pb", np.nan)
        turnover_rate = getattr(r, "turnover_rate", np.nan)
        volume_ratio_basic = getattr(r, "volume_ratio_basic", np.nan)

        try:
            # liquidity gate
            if is_main:
                if not (np.isfinite(amt_now) and amt_now >= MIN_AMOUNT_MAIN):
                    continue
            else:
                if not (np.isfinite(amt_now) and amt_now >= MIN_AMOUNT_SECOND):
                    continue

            hdf = get_hist_df(code, tail=TAIL)
            if hdf is None or hdf.empty:
                continue

            ohlc = pick_ohlc_columns(hdf)
            if not ohlc:
                continue
            ocol, ccol, hcol, lcol = ohlc

            close = pd.to_numeric(hdf[ccol], errors="coerce")
            high = pd.to_numeric(hdf[hcol], errors="coerce")
            low = pd.to_numeric(hdf[lcol], errors="coerce")
            # ✅ stable index for open_
            open_ = pd.to_numeric(hdf[ocol], errors="coerce") if ocol else pd.Series(np.nan, index=hdf.index)

            if close.isna().sum() > 5 or len(close) < LOOKBACK + 40:
                continue

            last = float(close.iloc[-1])
            if not np.isfinite(last) or last <= 0:
                continue

            ret_120 = last / float(close.iloc[-LOOKBACK]) - 1.0
            if not np.isfinite(ret_120) or ret_120 <= 0:
                continue

            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma60 = float(close.rolling(60).mean().iloc[-1])
            ma120 = float(close.rolling(120).mean().iloc[-1])

            ma_stack = int(ma20 > ma60 > ma120)
            above_ma60 = int(last >= ma60)

            atr_series = calc_atr_series_wilder(high, low, close, n=14)
            atr_now = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else np.nan
            atr_ratio = (atr_now / last) if (np.isfinite(atr_now) and last > 0) else np.nan
            if np.isfinite(atr_ratio) and atr_ratio >= ATR_HARD_MAX:
                continue
            atr_score = atr_penalty(atr_ratio, code)
            if atr_score <= 0:
                continue

            high20 = float(high.tail(20).max())
            high60 = float(high.tail(60).max())
            dist20 = (high20 - last) / high20 if high20 > 0 else np.nan
            dist60 = (high60 - last) / high60 if high60 > 0 else np.nan

            max_abs = breakout_gate_abs_pct(code, dist20)
            if np.isfinite(pct_now) and abs(pct_now) > max_abs:
                continue

            vr = calc_vol_ratio_from_hist(hdf)
            data_quality = "OK" if np.isfinite(vr) else "NO_VOL"
            tight10 = close_tightness_10(close)

            atr_ago = float(atr_series.iloc[-21]) if len(atr_series) >= 21 and pd.notna(atr_series.iloc[-21]) else np.nan
            trend = soft_trend_score(ret_120)
            trend_r2, max_dd = momentum_path_quality(close, LOOKBACK)
            pq = path_quality_score(trend_r2, max_dd)
            mom = 0.78 * trend + 0.22 * pq

            liq = math.log10(amt_now + 1) if np.isfinite(amt_now) and amt_now > 0 else 0.0
            liq = clamp((liq - 9.0) / 2.6, 0, 1)

            vrs_expand = vol_ratio_score_expand(vr)
            d20_expand = dist_to_high_score(dist20, vr, mode_expand=True)
            d60_expand = dist_to_high_score(dist60, vr, mode_expand=True)

            mad = ma_dynamics(close)
            ma_sl = ma_slope_score(mad["ma20_slope"])
            ma_cv = ma_converge_score(mad["ma_convergence"], ma20 >= ma60)

            vcp = 0.0 if ret_120 < VCP_MIN_RET120 else vcp_score(vr, dist20, last, ma60, tight10, atr_now, atr_ago)

            # Universe B must have structure
            if not is_main:
                pre_breakout = (np.isfinite(dist20) and dist20 <= 0.05 and np.isfinite(vr) and vr >= 1.2)
                if not (vcp > 0.0 or pre_breakout):
                    continue

            # penalties
            blow = blowoff_penalty(vr, pct_now, dist20, ret_120)
            chase = chase_penalty(pct_now, code)
            bias_p = bias20_penalty(last, ma20)

            # RSRS (✅ stable input)
            rsrs_s, rsrs_r2 = calc_rsrs_score(pd.DataFrame({"high": high.values, "low": low.values}), N=18, M=180)
            rsrs_conf = 1.0
            if np.isfinite(rsrs_r2) and rsrs_r2 < RSRS_MIN_R2:
                rsrs_conf = 0.90  # low r2 => down-weight

            # smart money (intraday strength) from last day
            sm = smart_money_factor(
                float(open_.iloc[-1]) if len(open_) else np.nan,
                last,
                float(high.iloc[-1]),
                float(low.iloc[-1]),
            )
            if np.isfinite(sm) and sm < SMART_MONEY_MIN:
                continue

            interact = interaction_bonus(dist20, vr, mad["ma_convergence"], ma20 >= ma60, tight10, mad["ma20_slope"])

            # model scores (enhanced but still stable)
            vcp_model = (0.50 * vcp + 0.16 * mom + 0.12 * atr_score + 0.08 * liq
                         + 0.08 * ma_cv + 0.06 * (0.85 if ma_stack else 0.45))
            vcp_model += interact["vcp_synergy"]

            breakout = (0.24 * d20_expand + 0.15 * d60_expand + 0.22 * vrs_expand + 0.16 * mom
                        + 0.09 * atr_score + 0.05 * liq + 0.05 * ma_sl + 0.04 * pq)
            breakout += interact["breakout_synergy"]

            early = (0.22 * mom + 0.16 * vrs_expand + 0.16 * atr_score + 0.12 * liq
                     + 0.10 * (0.8 if above_ma60 else 0.2) + 0.08 * (0.85 if ma20 >= ma60 else 0.45)
                     + 0.08 * ma_sl + 0.06 * ma_cv + 0.04 * (1.0 if ma_stack else 0.0))
            early += interact["trend_confirm"]

            cont = (0.22 * (0.95 if ma_stack else 0.45) + 0.18 * mom + 0.14 * d20_expand + 0.12 * vrs_expand
                    + 0.10 * atr_score + 0.05 * liq + 0.11 * ma_sl + 0.08 * ma_cv)
            if np.isfinite(dist20) and dist20 > 0.16:
                cont -= 0.10
            cont += interact["trend_confirm"]

            # regime multipliers (tiny)
            vcp_model *= w["vcp_mul"]
            breakout  *= w["breakout_mul"]
            early     *= w["early_mul"]
            cont      *= w["cont_mul"]

            # apply penalties + rsrs confidence
            def finalize(x: float) -> float:
                x = x * rsrs_conf
                x = x - blow - chase - bias_p
                return float(clamp(x, 0, 1.25))

            vcp_model = finalize(vcp_model)
            breakout  = finalize(breakout)
            early     = finalize(early)
            cont      = finalize(cont)

            scores = {"VCP_缩量收敛前夜": vcp_model, "BREAKOUT_即将启动": breakout, "EARLY_TREND_主线前期": early, "CONTINUATION_主升延续": cont}
            model = max(scores, key=scores.get)
            model_score = scores[model]
            final_score = max(scores.values())

            item = {
                "code": code, "name": name, "model": model,
                "price": round(price_now, 2) if np.isfinite(price_now) else np.nan,
                "pct": round(pct_now, 2) if np.isfinite(pct_now) else np.nan,
                "amount_yuan": amt_now,

                "ret_120": float(ret_120),
                "ret_120_pct": round(ret_120 * 100, 1),

                "atr_ratio": float(atr_ratio) if np.isfinite(atr_ratio) else np.nan,
                "atr_pct": round(atr_ratio * 100, 2) if np.isfinite(atr_ratio) else np.nan,

                "dist20": float(dist20) if np.isfinite(dist20) else np.nan,
                "dist_20d_high_pct": round(dist20 * 100, 2) if np.isfinite(dist20) else np.nan,

                "vol_ratio": float(vr) if np.isfinite(vr) else np.nan,
                "tight10": float(tight10) if np.isfinite(tight10) else np.nan,
                "tight10_pct": round(tight10 * 100, 2) if np.isfinite(tight10) else np.nan,

                "rsrs_score": float(rsrs_s) if np.isfinite(rsrs_s) else np.nan,
                "rsrs_r2": float(rsrs_r2) if np.isfinite(rsrs_r2) else np.nan,

                "smart_money": float(sm) if np.isfinite(sm) else np.nan,

                "pe_ttm": float(pe_ttm) if np.isfinite(pe_ttm) else np.nan,
                "pb": float(pb) if np.isfinite(pb) else np.nan,
                "turnover_rate": float(turnover_rate) if np.isfinite(turnover_rate) else np.nan,
                "volume_ratio_basic": float(volume_ratio_basic) if np.isfinite(volume_ratio_basic) else np.nan,

                "trend_r2": float(trend_r2) if np.isfinite(trend_r2) else np.nan,
                "max_dd": float(max_dd) if np.isfinite(max_dd) else np.nan,
                "path_quality": round(pq, 4),
                "ma20_slope": float(mad["ma20_slope"]) if np.isfinite(mad["ma20_slope"]) else np.nan,
                "ma_convergence": round(mad["ma_convergence"], 4) if np.isfinite(mad["ma_convergence"]) else np.nan,
                "ma20_above_days": int(mad["ma20_above_days"]),
                "interact_breakout": round(interact["breakout_synergy"], 3),
                "interact_vcp": round(interact["vcp_synergy"], 3),
                "interact_trend": round(interact["trend_confirm"], 3),

                "ma_stack": int(ma_stack),
                "above_ma60": int(above_ma60),

                "data_quality": data_quality,
                "blowoff_penalty": round(blow, 3),
                "chase_penalty": round(chase, 3),
                "bias20_penalty": round(bias_p, 3),

                "model_score": round(model_score, 4),
                "score": round(final_score, 4),
                "is_main_universe": int(is_main),
            }

            buckets[model].append(item)
            all_candidates.append(item)

        except Exception as e:
            fail += 1
            if fail <= 15:
                print(f"[fail] {code} {name}: {type(e).__name__}: {e}")

        if i % 100 == 0:
            print(f"progress {i}/{total} candidates={len(all_candidates)} fail={fail}")
        time.sleep(SLEEP)

    if not all_candidates:
        print("候选为空：降低 MIN_AMOUNT 或放宽参数。")
        return

    res = pd.DataFrame(all_candidates)

    # rank_score + fused
    res["rank_score"] = build_rank_score(res)
    res["score_fused"] = (FUSE_W_SCORE * pd.to_numeric(res["score"], errors="coerce") +
                          FUSE_W_RANK  * pd.to_numeric(res["rank_score"], errors="coerce"))

    # tradability tie-break: prefer smaller pct + closer to high
    res = res.sort_values(["score_fused", "pct", "dist20"], ascending=[False, True, True])

    print("\n================= 🎯 总榜 Top (Fused) =================")
    show_cols = [
        "code","name","model","price","pct","amount_yuan",
        "ret_120_pct","atr_pct","dist_20d_high_pct","vol_ratio","tight10_pct",
        "rsrs_score","rsrs_r2","smart_money","path_quality","ma20_slope","pe_ttm","turnover_rate",
        "blowoff_penalty","chase_penalty","bias20_penalty",
        "score","rank_score","score_fused"
    ]
    for c in show_cols:
        if c not in res.columns:
            res[c] = np.nan
    top_df = res.head(int(topk_out)).copy()
    print(top_df[show_cols].to_string(index=False))

    best = res.iloc[0].to_dict()
    print("\n✅ 我给你的“选一个”（最终融合评分最佳）")
    print(best)

    # charts
    if make_charts:
        save_charts(res)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regime": regime,
        "total_candidates": int(len(res)),
        "topk": int(len(top_df)),
        "best": best,
        "rows": top_df.to_dict(orient="records"),
    }


def main():
    run_selector(topk_out=TOPK_OUT, make_charts=True)


if __name__ == "__main__":
    main()
