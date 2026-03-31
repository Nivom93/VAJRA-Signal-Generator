#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_oracle.py — Asynchronous LLM Macro Oracle
==============================================
DIRECTIVE EXECUTED: Google Gemini SDK Migration (gemini-2.5-flash)
Fetches CoinDesk RSS news feed, extracts the 15 most recent article titles,
and uses Gemini to generate a single float sentiment score between -1.0 and 1.0.
"""

import urllib.request
import xml.etree.ElementTree as ET
import json
import time
import os
import logging
import re
from pathlib import Path

# Setup Logging
log = logging.getLogger("vajra.oracle")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

def fetch_rss_titles(url="https://www.coindesk.com/arc/outboundfeeds/rss/", limit=15):
    """Fetches the latest crypto headlines from CoinDesk."""
    try:
        # Standard user-agent to avoid basic anti-bot blocks
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        titles = []
        # Find all item elements, typically inside the channel element
        for item in root.findall('.//item'):
            title_elem = item.find('title')
            if title_elem is not None and title_elem.text:
                titles.append(title_elem.text.strip())
            if len(titles) >= limit:
                break
        return titles
    except Exception as e:
        log.error(f"Failed to fetch RSS feed: {e}")
        return []

def get_llm_sentiment(titles):
    """Passes headlines to Google Gemini to extract a precise quantitative sentiment score."""
    if not titles:
        return 0.0

    try:
        from google import genai
        if os.environ.get("GEMINI_API_KEY"):
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            prompt = (
                "You are an elite quantitative macro analyst. "
                "Read these recent crypto headlines. "
                "Output a SINGLE FLOAT between -1.0 (extreme bearish) and 1.0 (extreme bullish). "
                "Output NOTHING ELSE."
            )
            headlines_text = "\n".join([f"- {title}" for title in titles])
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt + "\n\n" + headlines_text
            )
            result_text = response.text.strip()
            sentiment_score = float(result_text)
            return max(-1.0, min(1.0, sentiment_score))

    except Exception as e:
        log.warning(f"Gemini API unavailable or failed ({e}). Falling back to Free Oracle...")

    # ==========================================================
    # FREE MACRO ORACLE FALLBACK (Alternative.me + Basic NLP)
    # ==========================================================
    log.info("Executing 100% FREE Macro Oracle Fallback (Fear & Greed + RSS NLP).")
    try:
        # 1. Alternative.me Crypto Fear & Greed Index
        fg_url = "https://api.alternative.me/fng/?limit=1"
        req = urllib.request.Request(fg_url, headers={'User-Agent': 'Mozilla/5.0'})
        fg_score = 0.0
        with urllib.request.urlopen(req, timeout=10) as response:
            fg_data = json.loads(response.read().decode('utf-8'))
            if 'data' in fg_data and len(fg_data['data']) > 0:
                raw_fg = float(fg_data['data'][0]['value']) # 0 to 100
                fg_score = (raw_fg - 50.0) / 50.0 # Map to -1.0 to 1.0

        # 2. Basic Keyword NLP on RSS Titles
        bull_words = {'surge', 'rally', 'bull', 'high', 'gain', 'jump', 'soar', 'approve', 'adoption', 'up', 'buy', 'long'}
        bear_words = {'crash', 'plunge', 'bear', 'low', 'loss', 'drop', 'dive', 'reject', 'ban', 'down', 'sell', 'short', 'hack', 'scam', 'sec'}

        nlp_score = 0.0
        if titles:
            bull_count = sum(1 for t in titles for w in bull_words if w in t.lower())
            bear_count = sum(1 for t in titles for w in bear_words if w in t.lower())
            total = bull_count + bear_count
            if total > 0:
                nlp_score = (bull_count - bear_count) / total

        # 3. Blended Sentiment (Weighted 70% F&G, 30% NLP)
        final_sentiment = (0.7 * fg_score) + (0.3 * nlp_score)
        return max(-1.0, min(1.0, final_sentiment))

    except Exception as fallback_e:
        log.error(f"Free Oracle Fallback completely failed: {fallback_e}")
        return 0.0

def main():
    log.info("Starting Vajra LLM Macro Oracle (Gemini 2.5 Flash Edition)...")
    cache_dir = Path("data_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "oracle_sentiment.json"

    while True:
        try:
            log.info("Fetching latest CoinDesk headlines...")
            titles = fetch_rss_titles()

            if not titles:
                log.warning("No titles fetched. Skipping LLM query.")
            else:
                log.info(f"Fetched {len(titles)} titles. Querying Gemini LLM...")
                sentiment = get_llm_sentiment(titles)

                log.info(f"🎯 Gemini Sentiment Score: {sentiment}")

                # Save to JSON Cache for Engine ingestion
                output_data = {
                    "timestamp": int(time.time() * 1000),
                    "sentiment": sentiment
                }

                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f)

                log.info(f"Saved sentiment to {cache_file}")

        except Exception as e:
            log.error(f"Oracle loop encountered a critical error: {e}")

        # Sleep for 30 minutes (1800 seconds) to conserve API limits
        log.info("Oracle sleeping for 30 minutes...")
        time.sleep(1800)

if __name__ == "__main__":
    main()
