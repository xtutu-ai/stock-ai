# service_ts.py
import os
import time
import math
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import tushare as ts
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from threading import RLock

import seelect5_enhanced

# =========================
# Config
# =========================
APP_TITLE = "Tushare A-Share Cache Service (Robust QFQ + Universe)"
CACHE_DIR = os.getenv("TS_CACHE_DIR", "./cache_ts")

# Files
BASIC_PARQUET = os.path.join(CACHE_DIR, "stock_basic.parquet")
UNIVERSE_PARQUET = os.path.join(CACHE_DIR, "universe_latest.parquet")
UNIVERSE_META_JSON = os.path.join(CACHE_DIR, "universe_meta.json")
SELECT_RESULT_JSON = os.path.join(CACHE_DIR, "selector_result_latest.json")

DAILY_DIR = os.path.join(CACHE_DIR, "daily")  # raw daily parquet per ts_code
ADJ_DIR = os.path.join(CACHE_DIR, "adj")      # adj_factor parquet per ts_code
QFQ_DIR = os.path.join(CACHE_DIR, "qfq")      # qfq parquet per ts_code

# Upstream throttling
UPSTREAM_MIN_INTERVAL_SEC = float(os.getenv("TS_UPSTREAM_MIN_INTERVAL_SEC", "0.45"))
_last_upstream_ts = 0.0
_upstream_lock = RLock()

# Lock pool to avoid _code_locks growth
LOCK_POOL_SIZE = int(os.getenv("TS_LOCK_POOL_SIZE", "256"))
_lock_pool = [RLock() for _ in range(LOCK_POOL_SIZE)]

def _code_lock(key: str) -> RLock:
    idx = (hash(key) % LOCK_POOL_SIZE)
    return _lock_pool[idx]

# Tushare token (MUST)

# global pro client (avoid re-init each request)
_pro_client = None
_pro_lock = RLock()

def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)
    os.makedirs(ADJ_DIR, exist_ok=True)
    os.makedirs(QFQ_DIR, exist_ok=True)

def _rate_limit_upstream():
    global _last_upstream_ts
    with _upstream_lock:
        now = time.time()
        gap = now - _last_upstream_ts
        if gap < UPSTREAM_MIN_INTERVAL_SEC:
            time.sleep(UPSTREAM_MIN_INTERVAL_SEC - gap)
        _last_upstream_ts = time.time()

def _pro():
    global _pro_client
    token = "5ab66f0e95fde5d02883b7e27056571a27ab72441fd897b9d13165d46380"
    _pro_client  = ts.pro_api(token)
    _pro_client._DataApi__token = token # 保证有这个代码，不然不可以获取
    _pro_client._DataApi__http_url = 'http://lianghua.nanyangqiankun.top' 
    return _pro_client

def _safe(ts_code: str) -> str:
    return ts_code.replace(".", "_")

def _daily_path(ts_code: str) -> str:
    return os.path.join(DAILY_DIR, f"{_safe(ts_code)}.parquet")

def _adj_path(ts_code: str) -> str:
    return os.path.join(ADJ_DIR, f"{_safe(ts_code)}.parquet")

def _qfq_path(ts_code: str) -> str:
    return os.path.join(QFQ_DIR, f"{_safe(ts_code)}.parquet")

def _read_parquet(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        # corrupted parquet etc.
        return None

def _write_parquet_atomic(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

def _merge_append(old_df: Optional[pd.DataFrame], new_df: pd.DataFrame, key: str) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return old_df if old_df is not None else pd.DataFrame()
    if old_df is None or old_df.empty:
        out = new_df.copy()
    else:
        out = pd.concat([old_df, new_df], ignore_index=True)
    out[key] = out[key].astype(str)
    out = out.drop_duplicates(subset=[key], keep="last").sort_values(key).reset_index(drop=True)
    return out

def _calc_qfq_from_daily(full_daily: pd.DataFrame, full_adj: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: use FULL adj_factor history; otherwise QFQ Trap will NaN out history.
    """
    if full_daily is None or full_daily.empty:
        return pd.DataFrame()

    d = full_daily.copy()
    d["trade_date"] = d["trade_date"].astype(str)

    # ensure columns
    for c in ["open", "high", "low", "close"]:
        if c not in d.columns:
            raise ValueError(f"daily missing column: {c}")

    if full_adj is None or full_adj.empty:
        raise ValueError("adj_factor is empty; cannot compute qfq")

    a = full_adj.copy()
    a["trade_date"] = a["trade_date"].astype(str)
    if "adj_factor" not in a.columns:
        raise ValueError("adj_factor missing column adj_factor")

    m = pd.merge(d, a[["trade_date", "adj_factor"]], on="trade_date", how="left")
    # if still missing, we cannot compute properly
    if m["adj_factor"].isna().sum() > 0:
        # allow small missing, but don’t destroy everything silently
        miss_ratio = m["adj_factor"].isna().mean()
        if miss_ratio > 0.02:
            raise ValueError(f"adj_factor missing too many rows: {miss_ratio:.2%}")

    # base factor: last non-null factor
    last_factor = m["adj_factor"].dropna().iloc[-1]
    ratio = m["adj_factor"] / last_factor

    out = pd.DataFrame({
        "ts_code": m["ts_code"],
        "trade_date": m["trade_date"],
        "open": pd.to_numeric(m["open"], errors="coerce") * ratio,
        "high": pd.to_numeric(m["high"], errors="coerce") * ratio,
        "low":  pd.to_numeric(m["low"], errors="coerce") * ratio,
        "close":pd.to_numeric(m["close"], errors="coerce") * ratio,
        # keep raw volume/amount (not adjusted)
        "vol": pd.to_numeric(m.get("vol", np.nan), errors="coerce"),
        "amount": pd.to_numeric(m.get("amount", np.nan), errors="coerce"),  # tushare: 千元
    })
    out["amount_yuan"] = out["amount"] * 1000.0
    return out.dropna(subset=["close"]).reset_index(drop=True)

def _load_universe_meta() -> Dict[str, Any]:
    if not os.path.exists(UNIVERSE_META_JSON):
        return {}
    try:
        return pd.read_json(UNIVERSE_META_JSON, typ="series").to_dict()
    except Exception:
        return {}

def _save_universe_meta(meta: Dict[str, Any]):
    s = pd.Series(meta)
    _write_parquet_atomic(pd.DataFrame([s.to_dict()]), UNIVERSE_META_JSON + ".parquet")
    # also write json for easy reading
    try:
        import json
        with open(UNIVERSE_META_JSON, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _ensure_stock_basic(force: bool = False) -> pd.DataFrame:
    _ensure_dir()
    if (not force) and os.path.exists(BASIC_PARQUET):
        df = pd.read_parquet(BASIC_PARQUET)
        if not df.empty:
            return df

    pro = _pro()
    _rate_limit_upstream()
    basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date"
    )
    if basic is None or basic.empty:
        raise HTTPException(status_code=500, detail="stock_basic empty from tushare")
    _write_parquet_atomic(basic, BASIC_PARQUET)
    return basic

# =========================
# FastAPI
# =========================
app = FastAPI(title=APP_TITLE)


def _today_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")


@app.get("/", response_class=HTMLResponse)
def home_page():
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tushare Universe + 选股看板</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; }
    button { border: none; border-radius: 8px; padding: 10px 14px; cursor: pointer; margin-right: 8px; }
    .primary { background: #2563eb; color: white; }
    .secondary { background: #f3f4f6; }
    .success { background: #059669; color: white; }
    #status { margin: 10px 0; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { text-align: left; }
    .meta { color: #6b7280; font-size: 13px; }
    h3 { margin-top: 30px; }
  </style>
</head>
<body>
  <h2>Tushare Universe + 选股看板</h2>
  <div>
    <button id="refreshBtn" class="primary">刷新今日Universe</button>
    <button id="reloadBtn" class="secondary">仅重新加载Universe</button>
    <button id="runSelectorBtn" class="success">运行选股脚本</button>
    <button id="reloadSelectorBtn" class="secondary">读取最近选股结果</button>
  </div>
  <p id="status">准备就绪</p>
  <p class="meta" id="meta"></p>

  <h3>Universe（前200）</h3>
  <table>
    <thead><tr><th>代码</th><th>名称</th><th>行业</th><th>价格</th><th>涨跌(%)</th><th>成交额(元)</th><th>换手率</th><th>市盈率TTM</th><th>市净率</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>

  <h3>选股结果（Top30）</h3>
  <p class="meta" id="selectorMeta"></p>
  <table>
    <thead><tr><th>代码</th><th>名称</th><th>模型</th><th>现价</th><th>涨跌(%)</th><th>成交额(元)</th><th>融合分</th><th>模型分</th><th>Rank分</th></tr></thead>
    <tbody id="selectorRows"></tbody>
  </table>

<script>
const statusEl = document.getElementById('status');
const rowsEl = document.getElementById('rows');
const metaEl = document.getElementById('meta');
const selectorRowsEl = document.getElementById('selectorRows');
const selectorMetaEl = document.getElementById('selectorMeta');

function fmt(v, digits = 2) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : '-';
}
function setStatus(msg, isError=false) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? '#dc2626' : '#111827';
}

async function loadUniverse(limit = 200) {
  try {
    const r = await fetch(`/universe?limit=${limit}`);
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    rowsEl.innerHTML = data.head.map((x) => `<tr><td>${x.code ?? ''}</td><td>${x.name ?? ''}</td><td>${x.industry ?? ''}</td><td>${fmt(x.price,2)}</td><td>${fmt(x.pct,2)}</td><td>${fmt(x.amount_yuan,0)}</td><td>${fmt(x.turnover_rate,2)}</td><td>${fmt(x.pe_ttm,2)}</td><td>${fmt(x.pb,2)}</td></tr>`).join('');
    metaEl.textContent = `trade_date=${data.meta?.trade_date ?? '-'} ｜ rows=${data.rows ?? 0} ｜ updated_at=${data.meta?.updated_at ?? '-'}`;
  } catch (e) {
    rowsEl.innerHTML = '';
    setStatus(`Universe加载失败：${e.message}`, true);
  }
}

async function loadSelectorResult() {
  try {
    const r = await fetch('/selector_result');
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    const rows = data.rows || [];
    selectorRowsEl.innerHTML = rows.map((x) => `<tr><td>${x.code ?? ''}</td><td>${x.name ?? ''}</td><td>${x.model ?? ''}</td><td>${fmt(x.price,2)}</td><td>${fmt(x.pct,2)}</td><td>${fmt(x.amount_yuan,0)}</td><td>${fmt(x.score_fused,4)}</td><td>${fmt(x.score,4)}</td><td>${fmt(x.rank_score,4)}</td></tr>`).join('');
    selectorMetaEl.textContent = `generated_at=${data.generated_at ?? '-'} ｜ regime=${data.regime ?? '-'} ｜ total_candidates=${data.total_candidates ?? 0} ｜ topk=${data.topk ?? rows.length}`;
    return rows.length;
  } catch (e) {
    selectorRowsEl.innerHTML = '';
    selectorMetaEl.textContent = `暂无选股结果：${e.message}`;
    return 0;
  }
}

async function refreshToday() {
  setStatus('正在刷新今日Universe...');
  try {
    const r = await fetch('/refresh_today?with_basic=1', { method: 'POST' });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    setStatus(`刷新成功：trade_date=${data.universe_meta?.trade_date}，rows=${data.rows}`);
    await loadUniverse();
  } catch (e) {
    setStatus(`刷新失败：${e.message}`, true);
  }
}

async function runSelector() {
  setStatus('正在运行选股脚本，请稍候（可能需要几十秒）...');
  try {
    const r = await fetch('/run_selector?topk=30', { method: 'POST' });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    setStatus(`选股完成：总候选=${data.total_candidates}，Top=${data.topk}`);
    await loadSelectorResult();
  } catch (e) {
    setStatus(`选股失败：${e.message}`, true);
  }
}

document.getElementById('refreshBtn').addEventListener('click', refreshToday);
document.getElementById('reloadBtn').addEventListener('click', () => loadUniverse());
document.getElementById('runSelectorBtn').addEventListener('click', runSelector);
document.getElementById('reloadSelectorBtn').addEventListener('click', loadSelectorResult);

loadUniverse();
loadSelectorResult();
</script>
</body>
</html>
"""

@app.get("/health")
def health():
    meta = _load_universe_meta()
    return {
        "ok": True,
        "cache_dir": os.path.abspath(CACHE_DIR),
        "universe_meta": meta,
        "upstream_min_interval_sec": UPSTREAM_MIN_INTERVAL_SEC,
    }

@app.post("/refresh_basic")
def refresh_basic():
    try:
        df = _ensure_stock_basic(force=True)
        return {
            "ok": True,
            "rows": int(len(df)),
            "saved_to": os.path.abspath(BASIC_PARQUET),
            "cache_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh_universe")
def refresh_universe(
    trade_date: str = Query(..., description="YYYYMMDD, must be a trading day"),
    with_basic: int = Query(1, description="1: merge daily_basic (pe/pb/turnover/volume_ratio); 0: skip"),
):
    """
    Build universe_latest.parquet for a given trade_date.
    - price = close
    - pct = pct_chg
    - amount_yuan = amount*1000 (tushare amount is 千元)
    - optional: merge daily_basic fields
    """
    _ensure_dir()
    pro = _pro()

    try:
        trade_date = str(trade_date).strip()
        if not (len(trade_date) == 8 and trade_date.isdigit()):
            raise ValueError("trade_date must be YYYYMMDD")

        # basic mapping
        basic = _ensure_stock_basic(force=False)
        basic_map = basic[["ts_code", "symbol", "name", "industry"]].copy()
        basic_map["code"] = basic_map["symbol"].astype(str).str.zfill(6)

        # daily for all
        _rate_limit_upstream()
        daily = pro.daily(trade_date=trade_date, fields="ts_code,trade_date,open,high,low,close,pct_chg,vol,amount")
        if daily is None or daily.empty:
            raise HTTPException(status_code=500, detail=f"pro.daily empty for trade_date={trade_date}")

        daily["amount_yuan"] = pd.to_numeric(daily["amount"], errors="coerce") * 1000.0
        daily["price"] = pd.to_numeric(daily["close"], errors="coerce")
        daily["pct"] = pd.to_numeric(daily["pct_chg"], errors="coerce")

        u = pd.merge(daily, basic_map, on="ts_code", how="left")

        # optional daily_basic
        if int(with_basic) == 1:
            _rate_limit_upstream()
            db = pro.daily_basic(
                trade_date=trade_date,
                fields="ts_code,trade_date,turnover_rate,pe_ttm,pb,volume_ratio,total_mv"
            )
            if db is not None and not db.empty:
                u = pd.merge(u, db, on=["ts_code", "trade_date"], how="left")
                u = u.rename(columns={"volume_ratio": "volume_ratio_basic"})
            else:
                u["turnover_rate"] = np.nan
                u["pe_ttm"] = np.nan
                u["pb"] = np.nan
                u["volume_ratio_basic"] = np.nan
                u["total_mv"] = np.nan
        else:
            u["turnover_rate"] = np.nan
            u["pe_ttm"] = np.nan
            u["pb"] = np.nan
            u["volume_ratio_basic"] = np.nan
            u["total_mv"] = np.nan

        # final columns for selector
        out = pd.DataFrame({
            "trade_date": u["trade_date"].astype(str),
            "code": u["code"].astype(str).str.zfill(6),
            "name": u["name"].astype(str),
            "ts_code": u["ts_code"].astype(str),
            "price": pd.to_numeric(u["price"], errors="coerce"),
            "pct": pd.to_numeric(u["pct"], errors="coerce"),
            "amount_yuan": pd.to_numeric(u["amount_yuan"], errors="coerce"),

            "pe_ttm": pd.to_numeric(u.get("pe_ttm", np.nan), errors="coerce"),
            "pb": pd.to_numeric(u.get("pb", np.nan), errors="coerce"),
            "turnover_rate": pd.to_numeric(u.get("turnover_rate", np.nan), errors="coerce"),
            "volume_ratio_basic": pd.to_numeric(u.get("volume_ratio_basic", np.nan), errors="coerce"),
            "total_mv": pd.to_numeric(u.get("total_mv", np.nan), errors="coerce"),
            "industry": u.get("industry", ""),
        })

        # remove invalid codes
        out = out[out["code"].str.match(r"^\d{6}$", na=False)].copy()

        _write_parquet_atomic(out, UNIVERSE_PARQUET)
        meta = {
            "trade_date": trade_date,
            "rows": int(len(out)),
            "with_basic": int(with_basic),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_universe_meta(meta)

        return {
            "ok": True,
            "rows": int(len(out)),
            "saved_to": os.path.abspath(UNIVERSE_PARQUET),
            "universe_meta": meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"refresh_universe failed: {e}")


@app.post("/refresh_today")
def refresh_today(with_basic: int = Query(1, description="same as /refresh_universe with_basic")):
    return refresh_universe(trade_date=_today_shanghai(), with_basic=with_basic)

@app.get("/universe")
def universe(limit: int = 200):
    if not os.path.exists(UNIVERSE_PARQUET):
        raise HTTPException(status_code=404, detail="universe not found. call /refresh_universe?trade_date=YYYYMMDD first.")
    df = pd.read_parquet(UNIVERSE_PARQUET)
    return {
        "ok": True,
        "rows": int(len(df)),
        "head": df.head(int(limit)).to_dict(orient="records"),
        "meta": _load_universe_meta(),
    }



@app.post("/run_selector")
def run_selector(topk: int = Query(30, ge=1, le=100)):
    _ensure_dir()
    try:
        result = seelect5_enhanced.run_selector(topk_out=topk, make_charts=False)
        if not isinstance(result, dict):
            raise RuntimeError("selector returned empty result")
        with open(SELECT_RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return {"ok": True, **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"run_selector failed: {e}")


@app.get("/selector_result")
def selector_result():
    if not os.path.exists(SELECT_RESULT_JSON):
        raise HTTPException(status_code=404, detail="selector result not found. call /run_selector first.")
    try:
        with open(SELECT_RESULT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, **data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load selector_result failed: {e}")

@app.post("/daily_refresh/{code}")
def daily_refresh(code: str, start_date: str = "20180101", end_date: str = ""):
    """
    Sync ONE stock daily + adj_factor and compute qfq parquet.
    code: 6-digit code (e.g., 300750)
    """
    _ensure_dir()
    code = str(code).strip().zfill(6)
    ts_code = None
    try:
        # map code -> ts_code
        basic = _ensure_stock_basic(force=False)
        row = basic[basic["symbol"].astype(str).str.zfill(6) == code]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"code {code} not found in stock_basic")
        ts_code = row.iloc[0]["ts_code"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"map ts_code failed: {e}")

    lock = _code_lock(ts_code)
    with lock:
        try:
            pro = _pro()
            if not end_date:
                end_date = time.strftime("%Y%m%d")

            # 1) fetch daily
            _rate_limit_upstream()
            new_daily = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,open,high,low,close,vol,amount"
            )

            if new_daily is None or new_daily.empty:
                # do not overwrite local cache with empty
                return {"ok": True, "ts_code": ts_code, "msg": "daily empty from tushare, skipped"}

            # 2) fetch adj_factor
            _rate_limit_upstream()
            new_adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if new_adj is None or new_adj.empty:
                return {"ok": True, "ts_code": ts_code, "msg": "adj_factor empty from tushare, skipped"}

            # 3) merge & persist daily
            d_path = _daily_path(ts_code)
            old_daily = _read_parquet(d_path)
            full_daily = _merge_append(old_daily, new_daily, "trade_date")
            _write_parquet_atomic(full_daily, d_path)

            # 4) merge & persist adj
            a_path = _adj_path(ts_code)
            old_adj = _read_parquet(a_path)
            full_adj = _merge_append(old_adj, new_adj, "trade_date")
            _write_parquet_atomic(full_adj, a_path)

            # 5) compute qfq using FULL adj history
            qfq_df = _calc_qfq_from_daily(full_daily, full_adj)
            q_path = _qfq_path(ts_code)
            _write_parquet_atomic(qfq_df, q_path)

            return {
                "ok": True,
                "code": code,
                "ts_code": ts_code,
                "daily_total": int(len(full_daily)),
                "adj_total": int(len(full_adj)),
                "qfq_total": int(len(qfq_df)),
                "saved": {
                    "daily": os.path.abspath(d_path),
                    "adj": os.path.abspath(a_path),
                    "qfq": os.path.abspath(q_path),
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"daily_refresh failed for {code} ({ts_code}): {type(e).__name__}: {e}")

@app.get("/hist/{code}")
def hist(code: str, tail: int = 280):
    """
    Return qfq history tail for selector.
    """
    code = str(code).strip().zfill(6)
    basic = _ensure_stock_basic(force=False)
    row = basic[basic["symbol"].astype(str).str.zfill(6) == code]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"code {code} not found")
    ts_code = row.iloc[0]["ts_code"]

    path = _qfq_path(ts_code)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"qfq not cached for {code}. call /daily_refresh/{code} first.")
    df = pd.read_parquet(path)
    if df.empty:
        raise HTTPException(status_code=404, detail="qfq empty")
    t = int(max(5, min(int(tail), len(df))))
    out = df.tail(t).copy()
    return {"ok": True, "code": code, "ts_code": ts_code, "tail": out.to_dict(orient="records")}
