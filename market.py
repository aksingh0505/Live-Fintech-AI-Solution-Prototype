# market_analysis.py

"""
Market Analysis Model

This file contains the logic for evaluating a company's financial health.
It uses mock data to simulate real-world financial metrics like
P/E ratio, revenue growth, and debt-to-equity ratio.

Enhancements:
- Contextual metrics derived from quarterly reports (YoY/QoQ growth, EPS surprise, guidance delta).
- Lightweight real-time update hooks to refresh quotes/metrics on intervals.
- Narrative context to explain why a score changed.

To plug in real data, replace the fetch_* functions with API calls
to providers like Financial Modeling Prep, Polygon, Alpha Vantage, or
SEC/EDGAR for filings. Keep return shapes unchanged for drop-in compatibility.
"""

import random
import time
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from contextlib import suppress

with suppress(Exception):
    import yfinance as yf  # optional live data source

logger = logging.getLogger("rtfe.market")

def _fetch_realtime_quote(ticker: str, use_live: bool = False) -> Dict[str, float]:
    """
    Fetch the latest real-time quote for a ticker.

    Replace the mock with an API call. Expected return keys:
    - price: last trade price
    - change_pct: percent change vs prior close
    - volume: latest cumulative volume
    """
    if not use_live:
        # Mock quote with small random walk
        base_prices = {"AAPL": 225.0, "MSFT": 435.0, "GOOG": 165.0}
        price = base_prices.get(ticker, 100.0) * (1 + random.uniform(-0.01, 0.01))
        return {
            "price": round(price, 2),
            "change_pct": round(random.uniform(-2.0, 2.0), 2),
            "volume": int(random.randint(1_000_000, 20_000_000)),
        }
    # Live path via yfinance if available
    try:
        t = yf.Ticker(ticker)  # type: ignore[name-defined]
        info = t.fast_info if hasattr(t, "fast_info") else {}
        price = float(getattr(info, "last_price", None) or info.get("last_price") or t.history(period="1d")["Close"].iloc[-1])
        prev_close = float(getattr(info, "previous_close", None) or info.get("previous_close") or price)
        change_pct = ((price - prev_close) / prev_close * 100.0) if prev_close else 0.0
        volume = int(getattr(info, "last_volume", None) or info.get("last_volume") or 0)
        return {"price": round(price, 2), "change_pct": round(change_pct, 2), "volume": volume}
    except Exception:
        logger.exception("yfinance quote failed for %s; falling back to mock.", ticker)
        return _fetch_realtime_quote(ticker, use_live=False)


def _fetch_quarterly_financials(ticker: str, use_live: bool = False) -> List[Dict[str, float]]:
    """
    Return the 4 most recent quarterly datapoints with consistent keys:
    - revenue, operating_income, net_income, eps
    - quarter_end (ISO date string)
    - eps_consensus (for surprise calc), guidance_revenue (next-q midpoint)
    """
    if not use_live:
        # Mock 4 quarters, newest first
        mock: Dict[str, List[Dict[str, float]]] = {
            "AAPL": [
                {"quarter_end": "2025-06-29", "revenue": 87000000000, "operating_income": 24000000000, "net_income": 20000000000, "eps": 1.43, "eps_consensus": 1.38, "guidance_revenue": 90000000000},
                {"quarter_end": "2025-03-30", "revenue": 82000000000, "operating_income": 22000000000, "net_income": 18500000000, "eps": 1.36, "eps_consensus": 1.31, "guidance_revenue": 85000000000},
                {"quarter_end": "2024-12-29", "revenue": 91000000000, "operating_income": 26000000000, "net_income": 21000000000, "eps": 1.50, "eps_consensus": 1.47, "guidance_revenue": 88000000000},
                {"quarter_end": "2024-09-29", "revenue": 76000000000, "operating_income": 19000000000, "net_income": 16000000000, "eps": 1.12, "eps_consensus": 1.10, "guidance_revenue": 80000000000},
            ],
            "MSFT": [
                {"quarter_end": "2025-06-30", "revenue": 68000000000, "operating_income": 29000000000, "net_income": 24000000000, "eps": 2.95, "eps_consensus": 2.85, "guidance_revenue": 70000000000},
                {"quarter_end": "2025-03-31", "revenue": 67000000000, "operating_income": 28000000000, "net_income": 23500000000, "eps": 2.88, "eps_consensus": 2.80, "guidance_revenue": 69000000000},
                {"quarter_end": "2024-12-31", "revenue": 65000000000, "operating_income": 27000000000, "net_income": 22800000000, "eps": 2.76, "eps_consensus": 2.70, "guidance_revenue": 66000000000},
                {"quarter_end": "2024-09-30", "revenue": 62000000000, "operating_income": 25000000000, "net_income": 21000000000, "eps": 2.55, "eps_consensus": 2.50, "guidance_revenue": 64000000000},
            ],
        }
        return mock.get(ticker, mock.get("AAPL"))
    # Live path via yfinance if available
    try:
        t = yf.Ticker(ticker)  # type: ignore[name-defined]
        # quarterly financials (income statement). yfinance returns DataFrame with columns as periods
        q_income = getattr(t, "quarterly_financials", None)
        quarters: List[Dict[str, float]] = []
        if q_income is not None and not q_income.empty:  # type: ignore[truthy-bool]
            # Take up to 4 most recent columns
            for col in list(q_income.columns)[:4]:
                try:
                    revenue = float(q_income.at["Total Revenue", col]) if "Total Revenue" in q_income.index else None
                    op_income = float(q_income.at["Operating Income", col]) if "Operating Income" in q_income.index else None
                    net_income = float(q_income.at["Net Income", col]) if "Net Income" in q_income.index else None
                    # EPS not directly available; approximate from net income over shares if possible
                    eps = None
                    quarter_end = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
                    if revenue and op_income and net_income:
                        quarters.append({
                            "quarter_end": quarter_end,
                            "revenue": revenue,
                            "operating_income": op_income,
                            "net_income": net_income,
                            "eps": eps if eps is not None else 1.0,  # placeholder
                            "eps_consensus": 1.0,
                            "guidance_revenue": revenue,  # placeholder
                        })
                except Exception:
                    continue
        return quarters or _fetch_quarterly_financials(ticker, use_live=False)
    except Exception:
        logger.exception("yfinance quarterlies failed for %s; falling back to mock.", ticker)
        return _fetch_quarterly_financials(ticker, use_live=False)


def _compute_growth(curr: float, prev: float) -> Optional[float]:
    if prev is None or prev == 0:
        return None
    return (curr - prev) / prev


def _build_quarterly_context(quarters: List[Dict[str, float]]) -> Dict[str, object]:
    """
    Compute QoQ and YoY growth, EPS surprise, and guidance deltas for the
    latest quarter and summarize trends across the last four quarters.
    """
    if not quarters:
        return {
            "latest_quarter": None,
            "qoq_growth": {},
            "yoy_growth": {},
            "eps_surprise_pct": None,
            "guidance_delta_pct": None,
            "trend_notes": [],
        }

    latest = quarters[0]
    prev_q = quarters[1] if len(quarters) > 1 else None
    # Use same quarter last year if available (index 3 for a 4-quarter list)
    prev_y = quarters[3] if len(quarters) > 3 else None

    qoq = {
        "revenue": _compute_growth(latest["revenue"], prev_q["revenue"]) if prev_q else None,
        "operating_income": _compute_growth(latest["operating_income"], prev_q["operating_income"]) if prev_q else None,
        "net_income": _compute_growth(latest["net_income"], prev_q["net_income"]) if prev_q else None,
        "eps": _compute_growth(latest["eps"], prev_q["eps"]) if prev_q else None,
    }

    yoy = {
        "revenue": _compute_growth(latest["revenue"], prev_y["revenue"]) if prev_y else None,
        "operating_income": _compute_growth(latest["operating_income"], prev_y["operating_income"]) if prev_y else None,
        "net_income": _compute_growth(latest["net_income"], prev_y["net_income"]) if prev_y else None,
        "eps": _compute_growth(latest["eps"], prev_y["eps"]) if prev_y else None,
    }

    eps_surprise_pct = None
    if latest.get("eps") is not None and latest.get("eps_consensus"):
        eps_surprise_pct = _compute_growth(latest["eps"], latest["eps_consensus"])  # (actual-cons)/cons

    guidance_delta_pct = None
    if latest.get("guidance_revenue") and prev_q and prev_q.get("revenue"):
        guidance_delta_pct = _compute_growth(latest["guidance_revenue"], prev_q["revenue"])  # guide vs last rev

    # Simple trend notes
    trend_notes: List[str] = []
    def _fmt_pct(x: Optional[float]) -> str:
        return f"{x*100:.1f}%" if isinstance(x, float) else "n/a"

    if isinstance(qoq.get("revenue"), float):
        note = "QoQ revenue accelerating" if qoq["revenue"] > 0 else "QoQ revenue contracting"
        trend_notes.append(f"{note} ({_fmt_pct(qoq['revenue'])}).")
    if isinstance(eps_surprise_pct, float):
        trend_notes.append(f"EPS surprise {_fmt_pct(eps_surprise_pct)} vs consensus.")
    if isinstance(guidance_delta_pct, float):
        ahead = "above" if guidance_delta_pct > 0 else "below"
        trend_notes.append(f"Next guidance {ahead} last quarter revenue by {_fmt_pct(guidance_delta_pct)}.")

    return {
        "latest_quarter": latest,
        "qoq_growth": qoq,
        "yoy_growth": yoy,
        "eps_surprise_pct": eps_surprise_pct,
        "guidance_delta_pct": guidance_delta_pct,
        "trend_notes": trend_notes,
    }


def _build_contextual_narrative(ticker: str, score: int, quarterly_ctx: Dict[str, object], quote: Dict[str, float]) -> List[str]:
    notes: List[str] = []
    change = quote.get("change_pct")
    if isinstance(change, float):
        if change > 0:
            notes.append(f"Shares are up {change:.2f}% today; momentum supports fundamentals.")
        elif change < 0:
            notes.append(f"Shares are down {abs(change):.2f}% today; watch for risk-off sentiment.")
    if quarterly_ctx.get("trend_notes"):
        notes.extend(quarterly_ctx["trend_notes"])  # type: ignore
    notes.append(f"Composite health score at {score}/100 based on valuation, growth, leverage.")
    return notes


def evaluate_financial_health(ticker: str, *, use_live: bool = False, include_context: bool = True) -> dict:
    """
    Evaluates a company's financial health based on periodic data.

    In a real-world scenario, this function would:
    1. Connect to a financial data API (e.g., Financial Modeling Prep, Polygon.io).
    2. Fetch the latest financial statements (Income Statement, Balance Sheet, Cash Flow).
    3. Calculate key financial ratios and metrics.

    For this basic implementation, we will use mock data.

    Args:
        ticker (str): The stock ticker of the company (e.g., 'AAPL', 'MSFT').

    Returns:
        dict: A dictionary containing the financial score and the metrics used.
              The score is on a scale of 0 to 100, where 100 is excellent health.
    """
    logger.info("Evaluating financial health for %s", ticker)

    # --- MOCK DATA SIMULATION ---
    # Replace this section with your actual API calls and data processing logic.
    mock_data = {
        'AAPL': {
            'pe_ratio': 28.5,
            'revenue_growth': 0.15,  # 15% growth
            'debt_to_equity': 1.2,
        },
        'MSFT': {
            'pe_ratio': 35.0,
            'revenue_growth': 0.20,  # 20% growth
            'debt_to_equity': 1.0,
        },
        'GOOG': {
            'pe_ratio': 25.0,
            'revenue_growth': 0.18,  # 18% growth
            'debt_to_equity': 0.8,
        },
    }

    financial_data = mock_data.get(ticker, None)

    if not financial_data:
        # If ticker is not in our mock data, generate random data
        logger.warning("Mock data not found for %s; generating random financial data.", ticker)
        financial_data = {
            'pe_ratio': random.uniform(15.0, 50.0),
            'revenue_growth': random.uniform(0.05, 0.30),
            'debt_to_equity': random.uniform(0.5, 2.0),
        }
    
    # --- SCORING LOGIC ---
    # Define a simple scoring algorithm. This is where you would implement
    # your proprietary model. Here, we use a weighted average of the metrics.
    
    # PE ratio score (lower is better, up to a point)
    pe_score = 100 - (financial_data['pe_ratio'] - 15) * 2 if financial_data['pe_ratio'] > 15 else 100
    
    # Revenue growth score (higher is better)
    revenue_score = min(100, financial_data['revenue_growth'] * 500)
    
    # Debt-to-equity score (lower is better)
    debt_score = 100 - (financial_data['debt_to_equity'] - 0.5) * 40
    
    # Ensure scores are within a valid range
    pe_score = max(0, min(100, pe_score))
    revenue_score = max(0, min(100, revenue_score))
    debt_score = max(0, min(100, debt_score))
    
    # Calculate the final composite score
    financial_score = (pe_score * 0.4) + (revenue_score * 0.4) + (debt_score * 0.2)
    financial_score = round(financial_score)

    # Contextual additions
    quarterly = _fetch_quarterly_financials(ticker, use_live=use_live)
    quarterly_context = _build_quarterly_context(quarterly)
    quote = _fetch_realtime_quote(ticker, use_live=use_live)

    # ------------------------------
    # Rigorous multi-quarter signals
    # ------------------------------
    def _ttm(values: List[float]) -> Optional[float]:
        if len(values) < 4:
            return None
        return float(sum(values[:4]))

    def _safe(values: List[Optional[float]]) -> List[float]:
        return [float(v) for v in values if isinstance(v, (int, float))]

    # Collect series newest first
    rev_series = _safe([q.get("revenue") if q else None for q in quarterly])
    op_series = _safe([q.get("operating_income") if q else None for q in quarterly])
    ni_series = _safe([q.get("net_income") if q else None for q in quarterly])
    eps_series = _safe([q.get("eps") if q else None for q in quarterly])

    ttm_rev = _ttm(rev_series) if rev_series else None
    ttm_op = _ttm(op_series) if op_series else None
    ttm_ni = _ttm(ni_series) if ni_series else None

    ttm_op_margin = (ttm_op / ttm_rev) if ttm_rev and ttm_rev != 0 and ttm_op is not None else None
    ttm_net_margin = (ttm_ni / ttm_rev) if ttm_rev and ttm_rev != 0 and ttm_ni is not None else None

    # Consistency: StdDev of QoQ revenue growth (lower is better)
    qoq_growth_series: List[float] = []
    for i in range(0, min(len(rev_series) - 1, 7)):
        prev = rev_series[i + 1]
        curr = rev_series[i]
        if prev:
            qoq_growth_series.append((curr - prev) / prev)
    growth_volatility = None
    if qoq_growth_series:
        mean = sum(qoq_growth_series) / len(qoq_growth_series)
        variance = sum((x - mean) ** 2 for x in qoq_growth_series) / len(qoq_growth_series)
        growth_volatility = variance ** 0.5

    # Margin trend: simple slope of operating margin over available quarters (normalized)
    margin_trend = None
    if len(rev_series) >= 2 and len(op_series) >= 2:
        margins: List[float] = []
        for i in range(min(len(rev_series), len(op_series))):
            r = rev_series[i]
            o = op_series[i]
            if r and r != 0:
                margins.append(o / r)
        if len(margins) >= 2:
            # slope via last-first over n
            margin_trend = (margins[0] - margins[-1]) / max(1, len(margins) - 1)

    # PEG-like: P/E divided by expected growth proxy (use latest YoY or average YoY)
    yoy = quarterly_context.get("yoy_growth", {}) if quarterly_context else {}
    yoy_rev = yoy.get("revenue") if isinstance(yoy, dict) else None
    expected_growth = None
    if isinstance(yoy_rev, float):
        expected_growth = max(0.01, min(0.50, yoy_rev))
    peg_like = None
    if expected_growth and financial_data.get("pe_ratio"):
        peg_like = float(financial_data["pe_ratio"]) / (expected_growth * 100)

    # Score adjustments based on rigorous signals (all on 0-100 scale contributions)
    rigorous_components: Dict[str, float] = {}

    if isinstance(ttm_op_margin, float):
        # 0% margin -> 30, 40% margin -> 100 (cap)
        rigorous_components["margin_quality"] = max(0.0, min(100.0, 30.0 + (ttm_op_margin * 175.0)))
    if isinstance(margin_trend, float):
        # positive trend boosts, negative penalizes; normalized around +/- 5pp per quarter
        rigorous_components["margin_trend"] = max(0.0, min(100.0, 50.0 + (margin_trend * 1000.0)))
    if isinstance(growth_volatility, float):
        # lower volatility better; 0.0 -> 100, 0.2 -> 40, 0.4 -> 0
        rigorous_components["growth_stability"] = max(0.0, min(100.0, 100.0 - (growth_volatility * 300.0)))
    if isinstance(peg_like, float):
        # PEG ~1 -> 70, <0.8 -> 90+, >2.0 -> 30
        rigorous_components["valuation_vs_growth"] = max(0.0, min(100.0, 110.0 - (peg_like * 40.0)))

    # Blend previous financial_score with rigorous overlay
    # base 50%, rigorous average 50%
    rigorous_avg = sum(rigorous_components.values()) / len(rigorous_components) if rigorous_components else financial_score
    final_score = int(round((financial_score * 0.5) + (rigorous_avg * 0.5)))

    result = {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "ticker": ticker,
        "score": final_score,
        "metrics": {
            **financial_data,
            # Extra inferred ratios for context
            "valuation_comment": "Lower P/E is generally better; growth can justify higher multiples.",
            "leverage_comment": "Lower debt-to-equity indicates less balance sheet risk.",
        },
        "quarterly_context": quarterly_context if include_context else None,
        "realtime_quote": quote if include_context else None,
        "rigorous_context": {
            "ttm_revenue": ttm_rev,
            "ttm_operating_income": ttm_op,
            "ttm_net_income": ttm_ni,
            "ttm_operating_margin": ttm_op_margin,
            "ttm_net_margin": ttm_net_margin,
            "growth_volatility": growth_volatility,
            "margin_trend_per_quarter": margin_trend,
            "peg_like": peg_like,
            "components": rigorous_components,
            "base_score": financial_score,
            "final_score": final_score,
        } if include_context else None,
    }

    if include_context:
        result["narrative_notes"] = _build_contextual_narrative(ticker, final_score, quarterly_context, quote)

    return result


def stream_realtime_updates(
    ticker: str,
    on_update: Callable[[dict], None],
    *,
    interval_seconds: int = 30,
    use_live: bool = False,
    max_updates: Optional[int] = None,
) -> None:
    """
    Periodically recompute the analysis and invoke a callback with the latest snapshot.

    - Replace polling with websockets if your data provider supports it.
    - Keep callback signature stable for UI consumers.
    """
    logger.info("Starting real-time stream for %s (every %ss)...", ticker, interval_seconds)
    count = 0
    while True:
        try:
            snapshot = evaluate_financial_health(ticker, use_live=use_live, include_context=True)
            on_update(snapshot)
        except Exception:
            logger.exception("Realtime update failed for %s; continuing loop.", ticker)
        count += 1
        if max_updates is not None and count >= max_updates:
            logger.info("Stream completed for %s", ticker)
            break
        time.sleep(max(5, interval_seconds))

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate financial health for a stock ticker (live via yfinance when --live).")
    parser.add_argument("--ticker", "-t", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    parser.add_argument("--live", action="store_true", help="Use free live sources (yfinance) when available")
    args = parser.parse_args()

    result = evaluate_financial_health(args.ticker, use_live=bool(args.live), include_context=True)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n--- Financial Health Report ---")
        print(f"Company: {result['ticker']}")
        print(f"As of: {result['as_of']}")
        print(f"Financial Score: {result['score']}/100")
        print("Metrics:")
        for key, value in result['metrics'].items():
            print(f"  - {key}: {value}")
        if result.get("realtime_quote"):
            q = result["realtime_quote"]
            print(f"Realtime: price={q.get('price')} change_pct={q.get('change_pct')}% volume={q.get('volume')}")
        if result.get("narrative_notes"):
            print("Notes:")
            for note in result["narrative_notes"]:
                print(f"  - {note}")
