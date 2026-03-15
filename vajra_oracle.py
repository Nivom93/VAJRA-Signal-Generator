#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_oracle.py — Asynchronous LLM Macro Oracle
==============================================
Fetches CoinDesk RSS news feed, extracts the 15 most recent article titles,
and uses an LLM to generate a single float sentiment score between -1.0 and 1.0.
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
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

def fetch_rss_titles(url="https://www.coindesk.com/arc/outboundfeeds/rss/", limit=15):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
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
    try:
        import openai

        # Ensure OPENAI_API_KEY is available in the environment
        if not os.environ.get("OPENAI_API_KEY"):
            log.warning("OPENAI_API_KEY not found. Returning neutral sentiment (0.0).")
            return 0.0

        client = openai.OpenAI()

        prompt = (
            "You are an elite quantitative macro analyst. "
            "Read these recent crypto headlines. "
            "Output a SINGLE FLOAT between -1.0 (extreme bearish) and 1.0 (extreme bullish). "
            "Output NOTHING ELSE."
        )

        headlines_text = "\n".join([f"- {title}" for title in titles])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": headlines_text}
            ],
            temperature=0.0,
            max_tokens=10
        )

        result_text = response.choices[0].message.content.strip()

        # Parse the float response safely with regex
        match = re.search(r"[-+]?\d*\.\d+|\d+", result_text)
        sentiment_score = float(match.group()) if match else 0.0

        # Ensure it's bounded correctly
        return max(-1.0, min(1.0, sentiment_score))

    except Exception as e:
        log.error(f"Failed to get LLM sentiment: {e}")
        return 0.0

def main():
    log.info("Starting Vajra LLM Macro Oracle...")
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
                log.info(f"Fetched {len(titles)} titles. Querying LLM...")
                sentiment = get_llm_sentiment(titles)

                log.info(f"LLM Sentiment Score: {sentiment}")

                # Save to JSON
                output_data = {
                    "timestamp": int(time.time() * 1000),
                    "sentiment": sentiment
                }

                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f)

                log.info(f"Saved sentiment to {cache_file}")

        except Exception as e:
            log.error(f"Oracle loop encountered an error: {e}")

        # Sleep for 4 hours
        log.info("Oracle sleeping for 4 hours...")
        time.sleep(14400)

if __name__ == "__main__":
    main()
