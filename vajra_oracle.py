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
        import google.generativeai as genai
    except ImportError:
        log.error("google-generativeai package not found. Install via: pip install google-generativeai")
        return 0.0

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            log.warning("GEMINI_API_KEY not found in environment. Returning neutral sentiment (0.0).")
            return 0.0

        # Configure Google Gemini SDK
        genai.configure(api_key=api_key)

        # Initialize the specific high-speed reasoning model
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Construct Prompt Geometry
        system_instructions = (
            "You are an elite quantitative macro analyst. "
            "Read these recent crypto headlines. "
            "Output a SINGLE FLOAT between -1.0 (extreme bearish) and 1.0 (extreme bullish). "
            "Output NOTHING ELSE. Do not include markdown, text, or explanation. Just the number."
        )

        headlines_text = "\n".join([f"- {title}" for title in titles])
        full_prompt = system_instructions + "\n\n" + headlines_text

        # Generate Content securely with strict algorithmic parameters
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=10,
            )
        )

        result_text = response.text.strip()
        log.debug(f"Raw Gemini Response: {result_text}")

        # Safely extract the float using Regex to prevent hallucination crashes
        match = re.search(r"[-+]?\d*\.\d+|\d+", result_text)
        if match:
            sentiment_score = float(match.group())
            # Ensure strictly mathematically bounded between -1.0 and 1.0
            return max(-1.0, min(1.0, sentiment_score))
        else:
            log.warning(f"Could not parse float from Gemini response: {result_text}")
            return 0.0

    except Exception as e:
        log.error(f"Failed to execute Gemini LLM sentiment extraction: {e}")
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

        # Sleep for 4 hours (14400 seconds) to conserve API limits
        log.info("Oracle sleeping for 4 hours...")
        time.sleep(14400)

if __name__ == "__main__":
    main()
