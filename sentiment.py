# sentiment_analysis.py

"""
Sentiment Analysis Model

This file contains the logic for analyzing media sentiment.
It uses the TextBlob library to process text data and generate
a sentiment score.

Enhancements:
- Adds contextual linkage to financial events (earnings, guidance, product launches).
- Provides a real-time polling helper that can stream updates to a callback.
- Returns per-headline metadata (time, source placeholder) for richer UIs.

Risk Factors Additions:
- Social media fetch mock with engagement metadata.
- Composite media + social risk score (0-100; higher = more risk).
- Topic flags for controversy/regulatory/operational risks and streaming updates.
"""

from textblob import TextBlob
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional
import random
import time
import logging
from urllib.parse import quote_plus
import json

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
import urllib.request  # type: ignore

logger = logging.getLogger("rtfe.sentiment")

# To use TextBlob for the first time, you need to download its corpora.
# Run the following commands in your terminal:
# pip install textblob
# python -m textblob.download_corpora

def _fetch_recent_headlines(ticker: str, use_live: bool = False) -> List[Dict[str, str]]:
    """
    Fetch recent headlines for a ticker. Replace mock with a news API.
    Return items with keys: title, published_at (ISO), source.
    """
    if not use_live:
        now = datetime.utcnow()
        return [
            {"title": f"Positive Q3 earnings report for {ticker} drives stock price up.", "published_at": (now - timedelta(minutes=35)).isoformat() + "Z", "source": "MockWire"},
            {"title": f"New product launch from {ticker} receives mixed reviews from tech experts.", "published_at": (now - timedelta(hours=2)).isoformat() + "Z", "source": "TechMock"},
            {"title": f"Analysts downgrade {ticker} due to supply chain concerns.", "published_at": (now - timedelta(hours=6)).isoformat() + "Z", "source": "StreetMock"},
            {"title": f"Innovative R&D team at {ticker} wins prestigious award.", "published_at": (now - timedelta(days=1)).isoformat() + "Z", "source": "AwardsDaily"},
            {"title": f"{ticker} announces a new partnership to expand into emerging markets.", "published_at": (now - timedelta(days=1, hours=3)).isoformat() + "Z", "source": "BizMock"},
            {"title": f"A brief overview of {ticker}'s performance this quarter.", "published_at": (now - timedelta(days=2)).isoformat() + "Z", "source": "MarketDaily"},
        ]
    # Live path via Google News RSS (no key). Intended for server-side use.
    try:
        query = quote_plus(f"{ticker} stock")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        content: bytes
        if requests is not None:
            resp = requests.get(rss_url, timeout=10)
            resp.raise_for_status()
            content = resp.content
        else:
            with urllib.request.urlopen(rss_url, timeout=10) as r:  # type: ignore
                content = r.read()
        text = content.decode("utf-8", errors="ignore")
        items: List[Dict[str, str]] = []
        for chunk in text.split("<item>")[1:]:
            try:
                title = chunk.split("<title>")[1].split("</title>")[0]
                pub = chunk.split("<pubDate>")[1].split("</pubDate>")[0]
            except Exception:
                continue
            items.append({"title": title, "published_at": pub, "source": "GoogleNews"})
            if len(items) >= 10:
                break
        return items or _fetch_recent_headlines(ticker, use_live=False)
    except Exception:
        logger.exception("Headlines fetch failed for %s; falling back to mock.", ticker)
        return _fetch_recent_headlines(ticker, use_live=False)


def _classify_context(title: str) -> str:
    t = title.lower()
    if "earnings" in t or "q" in t and "report" in t:
        return "earnings"
    if "guidance" in t:
        return "guidance"
    if "downgrade" in t or "upgrade" in t:
        return "analyst_rating"
    if "launch" in t or "partnership" in t:
        return "strategic"
    return "general"


def analyze_media_sentiment(ticker: str, *, use_live: bool = False, include_context: bool = True) -> dict:
    """
    Analyzes media sentiment for a given company ticker.

    In a real-world scenario, this function would:
    1. Connect to a news API (e.g., NewsAPI, GNews).
    2. Fetch recent news headlines and articles for the given ticker.
    3. Process the text to derive an overall sentiment score.

    For this basic implementation, we will use mock news headlines.

    Args:
        ticker (str): The stock ticker of the company (e.g., 'AAPL', 'MSFT').

    Returns:
        dict: A dictionary containing the average sentiment score and the headlines analyzed.
              The score is a floating-point value between -1.0 (very negative) and 1.0 (very positive).
    """
    logger.info("Analyzing media sentiment for %s", ticker)

    # Fetch headlines
    items = _fetch_recent_headlines(ticker, use_live=use_live)

    enriched: List[Dict[str, object]] = []
    sentiments: List[float] = []
    for item in items:
        title = item["title"]
        try:
            analysis = TextBlob(title)
            polarity = float(analysis.sentiment.polarity)
        except Exception:
            logger.exception("TextBlob failed on headline for %s; defaulting polarity to 0.0", ticker)
            polarity = 0.0
        context = _classify_context(title)
        sentiments.append(polarity)
        enriched.append({
            "title": title,
            "published_at": item.get("published_at"),
            "source": item.get("source"),
            "polarity": polarity,
            "context": context,
        })
        logger.debug("Headline: %s | Sentiment: %.2f | Context: %s", title, polarity, context)

    avg = sum(sentiments) / len(sentiments) if sentiments else 0.0

    # Light narrative context
    buckets = {"earnings": [], "guidance": [], "analyst_rating": [], "strategic": [], "general": []}
    for h in enriched:
        buckets[h["context"]].append(h)
    narrative_notes: List[str] = []
    if buckets["earnings"]:
        narrative_notes.append("Earnings-related coverage present; consider EPS surprise in fundamentals.")
    if buckets["guidance"]:
        narrative_notes.append("Guidance-focused headlines may drive forward-looking sentiment.")
    if buckets["analyst_rating"]:
        narrative_notes.append("Analyst rating changes detected; short-term volatility likely.")
    if buckets["strategic"]:
        narrative_notes.append("Strategic news (launches/partnerships) could shift growth narrative.")

    result = {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "ticker": ticker,
        "average_score": avg,
        "headlines": enriched if include_context else None,
        "narrative_notes": narrative_notes if include_context else None,
    }
    return result


def stream_sentiment_updates(
    ticker: str,
    on_update: Callable[[dict], None],
    *,
    interval_seconds: int = 60,
    use_live: bool = False,
    max_updates: Optional[int] = None,
) -> None:
    """
    Periodically fetch headlines and recompute sentiment, invoking callback with updates.
    Replace polling with provider webhooks or sockets when available.
    """
    logger.info("Starting sentiment stream for %s (every %ss)...", ticker, interval_seconds)
    count = 0
    while True:
        try:
            snapshot = analyze_media_sentiment(ticker, use_live=use_live, include_context=True)
            on_update(snapshot)
        except Exception:
            logger.exception("Sentiment update failed for %s; continuing loop.", ticker)
        count += 1
        if max_updates is not None and count >= max_updates:
            logger.info("Sentiment stream completed for %s", ticker)
            break
        time.sleep(max(5, interval_seconds))


# ----------------
# Risk computation
# ----------------

def _fetch_social_posts(ticker: str, use_live: bool = False) -> List[Dict[str, object]]:
    """
    Fetch recent social posts mentioning the ticker. Replace with live API(s):
    X/Twitter, Reddit, StockTwits, etc.

    Returns list items with keys:
    - text: content string
    - published_at: ISO timestamp
    - platform: str
    - engagement: dict with likes/replies/shares
    - author_followers: int (approx reach)
    """
    if not use_live:
        now = datetime.utcnow()
        def mk(text: str, minutes_ago: int, platform: str, likes: int, replies: int, shares: int, followers: int) -> Dict[str, object]:
            return {
                "text": text,
                "published_at": (now - timedelta(minutes=minutes_ago)).isoformat() + "Z",
                "platform": platform,
                "engagement": {"likes": likes, "replies": replies, "shares": shares},
                "author_followers": followers,
            }
        return [
            mk(f"{ticker} earnings beat but guidance light; sell the rip?", 20, "twitter", 320, 95, 110, 50000),
            mk(f"Big {ticker} outage reported by users across EU region.", 45, "reddit", 210, 140, 60, 15000),
            mk(f"Hearing {ticker} faces potential regulatory scrutiny over acquisition.", 75, "twitter", 410, 180, 200, 120000),
            mk(f"Loving the new {ticker} feature rollout. Smooth UX.", 130, "twitter", 85, 12, 9, 8000),
            mk(f"Rumors of layoffs at {ticker} next quarter.", 200, "stocktwits", 150, 60, 30, 10000),
        ]
    # Live path via Reddit search JSON (no key). Intended for server-side use.
    try:
        q = quote_plus(ticker)
        url = f"https://www.reddit.com/search.json?q={q}&sort=new&limit=10"
        headers = {"User-Agent": "rtfe/1.0"}
        data: dict
        if requests is not None:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        else:
            req = urllib.request.Request(url, headers=headers)  # type: ignore
            with urllib.request.urlopen(req, timeout=10) as r:  # type: ignore
                data = json.loads(r.read().decode("utf-8", errors="ignore"))
        children = data.get("data", {}).get("children", [])
        out: List[Dict[str, object]] = []
        for c in children:
            pr = c.get("data", {})
            title = pr.get("title") or pr.get("selftext") or ""
            created_utc = pr.get("created_utc")
            published_at = datetime.utcfromtimestamp(created_utc).isoformat() + "Z" if created_utc else datetime.utcnow().isoformat() + "Z"
            score = int(pr.get("score", 0))
            num_comments = int(pr.get("num_comments", 0))
            out.append({
                "text": title,
                "published_at": published_at,
                "platform": "reddit",
                "engagement": {"likes": score, "replies": num_comments, "shares": 0},
                "author_followers": 0,
            })
        return out or _fetch_social_posts(ticker, use_live=False)
    except Exception:
        logger.exception("Reddit fetch failed for %s; falling back to mock.", ticker)
        return _fetch_social_posts(ticker, use_live=False)


def _polarity(text: str) -> float:
    return float(TextBlob(text).sentiment.polarity)


def _contains_topics(text: str, topics: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in topics)


def _compute_social_volume_index(posts: List[Dict[str, object]]) -> float:
    """
    A simple heuristic combining count, engagement, and reach.
    Baseline ~1.0. >1 implies elevated chatter.
    """
    if not posts:
        return 0.0
    count_score = len(posts) / 10.0
    engagement_score = sum(int(p.get("engagement", {}).get("likes", 0)) + int(p.get("engagement", {}).get("replies", 0)) + int(p.get("engagement", {}).get("shares", 0)) for p in posts) / 1000.0
    reach_score = sum(int(p.get("author_followers", 0)) for p in posts) / 500_000.0
    return max(0.0, count_score + engagement_score + reach_score)


def analyze_risk_factors(ticker: str, *, use_live: bool = False) -> dict:
    """
    Combine media sentiment and social chatter into a composite risk view.
    Returns a dict with risk_score (0-100; higher means higher risk),
    components, topic flags, and narrative.
    """
    media = analyze_media_sentiment(ticker, use_live=use_live, include_context=True)
    posts = _fetch_social_posts(ticker, use_live=use_live)

    # Social sentiment distribution
    social_scores: List[float] = []
    negative_posts: List[Dict[str, object]] = []
    controversy_keywords = ["lawsuit", "probe", "scrutiny", "regulator", "antitrust", "outage", "breach", "recall", "layoff", "guidance light", "miss"]
    controversy_flags: List[str] = []

    for p in posts:
        text = str(p.get("text", ""))
        pol = _polarity(text)
        social_scores.append(pol)
        if pol < -0.15:
            negative_posts.append(p)
        if _contains_topics(text, controversy_keywords):
            # append matched keyword(s) for transparency (first match only for brevity)
            for k in controversy_keywords:
                if k in text.lower():
                    controversy_flags.append(k)
                    break

    social_avg = sum(social_scores) / len(social_scores) if social_scores else 0.0
    social_negative_ratio = (len([s for s in social_scores if s < 0]) / len(social_scores)) if social_scores else 0.0
    social_volume_index = _compute_social_volume_index(posts)

    # Media negativity proxy
    media_avg = float(media.get("average_score", 0.0))
    media_items = media.get("headlines") or []
    media_negative_ratio = 0.0
    if media_items:
        media_negative_ratio = len([h for h in media_items if float(h.get("polarity", 0.0)) < 0]) / max(1, len(media_items))

    # Composite risk scoring heuristic
    # Factors (weights sum to 1): media_neg (0.35), social_neg (0.35), volume_spike (0.2), controversy (0.1)
    volume_spike_component = min(1.0, social_volume_index)  # cap at 1
    controversy_component = min(1.0, len(set(controversy_flags)) / 3.0)

    # Convert avg sentiments to risk (invert sign; more negative => higher risk)
    media_neg_component = (media_negative_ratio * 0.7) + max(0.0, (-media_avg)) * 0.3
    social_neg_component = (social_negative_ratio * 0.6) + max(0.0, (-social_avg)) * 0.4

    composite_0_1 = (
        0.35 * media_neg_component +
        0.35 * social_neg_component +
        0.20 * volume_spike_component +
        0.10 * controversy_component
    )
    risk_score = int(round(max(0.0, min(1.0, composite_0_1)) * 100))

    # Narrative
    notes: List[str] = []
    if risk_score >= 70:
        notes.append("Elevated risk driven by negative tone and/or chatter spike.")
    elif risk_score >= 40:
        notes.append("Moderate risk; monitor sentiment momentum and topics.")
    else:
        notes.append("Low immediate sentiment-driven risk, barring new headlines.")
    if volume_spike_component > 0.6:
        notes.append("Social volume spike suggests higher short-term volatility.")
    if controversy_flags:
        notes.append("Controversy topics detected: " + ", ".join(sorted(set(controversy_flags))) + ".")
    if media.get("narrative_notes"):
        notes.extend(media["narrative_notes"])  # type: ignore

    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "ticker": ticker,
        "risk_score": risk_score,
        "components": {
            "media_negative_ratio": media_negative_ratio,
            "media_avg": media_avg,
            "social_negative_ratio": social_negative_ratio,
            "social_avg": social_avg,
            "social_volume_index": social_volume_index,
            "controversy_intensity": controversy_component,
        },
        "media": media,
        "social_posts": posts,
        "topic_flags": sorted(list(set(controversy_flags))),
        "narrative_notes": notes,
    }


def stream_risk_updates(
    ticker: str,
    on_update: Callable[[dict], None],
    *,
    interval_seconds: int = 60,
    use_live: bool = False,
    max_updates: Optional[int] = None,
) -> None:
    logger.info("Starting risk stream for %s (every %ss)...", ticker, interval_seconds)
    count = 0
    while True:
        try:
            snapshot = analyze_risk_factors(ticker, use_live=use_live)
            on_update(snapshot)
        except Exception:
            logger.exception("Risk update failed for %s; continuing loop.", ticker)
        count += 1
        if max_updates is not None and count >= max_updates:
            logger.info("Risk stream completed for %s", ticker)
            break
        time.sleep(max(5, interval_seconds))

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze media sentiment and optional risk for a ticker (live when --live).")
    parser.add_argument("--ticker", "-t", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--risk", action="store_true", help="Include composite risk analysis")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    parser.add_argument("--live", action="store_true", help="Use free live sources (Google News RSS, Reddit)")
    args = parser.parse_args()

    sentiment = analyze_media_sentiment(args.ticker, use_live=bool(args.live), include_context=True)
    result = {"sentiment": sentiment}
    if args.risk:
        risk = analyze_risk_factors(args.ticker, use_live=bool(args.live))
        result["risk"] = risk

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n--- Media Sentiment Report ---")
        print(f"Company: {sentiment['ticker']}")
        print(f"As of: {sentiment['as_of']}")
        print(f"Average Sentiment Score: {sentiment['average_score']:.2f}")
        if args.risk:
            r = result["risk"]
            print("\n--- Risk Factors ---")
            print(f"Risk Score: {r['risk_score']} / 100")
            print("Key flags:", ", ".join(r.get("topic_flags", [])) or "None")
            if r.get("narrative_notes"):
                print("Notes:")
                for note in r["narrative_notes"]:
                    print(f"  - {note}")
