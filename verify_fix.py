import os

with open('vajra_live.py', 'r') as f:
    data = f.read()

# 1. Update _fetch_macro_context variables and return
data = data.replace('''def _fetch_macro_context(ex, global_cache, cfg):
    btc_bullish = 1.0
    btcd_trend = 0.0
    dxy_val = 0.0
    spx_val = 0.0
    oracle_sentiment_val = 0.0

    try:''', '''def _fetch_macro_context(ex, global_cache, cfg):
    btc_bullish = 1.0
    btcd_trend = 0.0
    oracle_sentiment_val = 0.0

    try:''')

data = data.replace('''        if getattr(cfg, 'use_macro_data', False):
            try:
                import yfinance as yf
                dxy_c = yf.Ticker("DX-Y.NYB").history(period="5d")['Close']
                spx_c = yf.Ticker("^GSPC").history(period="5d")['Close']
                dxy_val = float(dxy_c.iloc[-1]) if not dxy_c.empty else 0.0
                spx_val = float(spx_c.iloc[-1]) if not spx_c.empty else 0.0
            except: pass

    except Exception as e:
        log.warning(f"Macro Sync Failed: {e}")''', '''    except Exception as e:
        log.warning(f"Macro Sync Failed: {e}")''')

data = data.replace('''    except Exception as e:
        log.warning(f"Oracle Sentiment Sync Failed: {e}")

    return btc_bullish, btcd_trend, dxy_val, spx_val, oracle_sentiment_val''', '''    except Exception as e:
        log.warning(f"Oracle Sentiment Sync Failed: {e}")

    return btc_bullish, btcd_trend, 0.0, 0.0, oracle_sentiment_val''')

data = data.replace('''            btc_bullish, btcd_trend, dxy_val, spx_val, oracle_sentiment_val = _fetch_macro_context(ex, global_cache, cfg)
            btc_ltf = global_cache.get("btc_ltf", pd.DataFrame())''', '''            btc_bullish, btcd_trend, _, _, oracle_sentiment_val = _fetch_macro_context(ex, global_cache, cfg)
            dxy_val = macro_fetcher.state["dxy_val"]
            spx_val = macro_fetcher.state["spx_val"]
            btc_ltf = global_cache.get("btc_ltf", pd.DataFrame())''')

with open('vajra_live.py', 'w') as f:
    f.write(data)
