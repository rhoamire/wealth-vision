"""
Multi Stock Exchange Data Handler
Supports: NSE, BSE, NYSE, NASDAQ, LSE
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# EXCHANGE CONFIGURATIONS
# ============================================================================

EXCHANGE_CONFIG = {
    "NSE": {
        "suffix": ".NS",
        "currency": "INR",
        "currency_symbol": "₹",
        "name": "National Stock Exchange (India)",
        "timezone": "Asia/Kolkata",
        "trading_hours": "09:15 - 15:30 IST",
        "index": "^NSEI",
        "index_name": "NIFTY 50",
    },
    "BSE": {
        "suffix": ".BO",
        "currency": "INR",
        "currency_symbol": "₹",
        "name": "Bombay Stock Exchange (India)",
        "timezone": "Asia/Kolkata",
        "trading_hours": "09:15 - 15:30 IST",
        "index": "^BSESN",
        "index_name": "SENSEX",
    },
    "NYSE": {
        "suffix": "",
        "currency": "USD",
        "currency_symbol": "$",
        "name": "New York Stock Exchange (US)",
        "timezone": "America/New_York",
        "trading_hours": "09:30 - 16:00 ET",
        "index": "^GSPC",
        "index_name": "S&P 500",
    },
    "NASDAQ": {
        "suffix": "",
        "currency": "USD",
        "currency_symbol": "$",
        "name": "NASDAQ (US)",
        "timezone": "America/New_York",
        "trading_hours": "09:30 - 16:00 ET",
        "index": "^IXIC",
        "index_name": "NASDAQ Composite",
    },
    "LSE": {
        "suffix": ".L",
        "currency": "GBP",
        "currency_symbol": "£",
        "name": "London Stock Exchange (UK)",
        "timezone": "Europe/London",
        "trading_hours": "08:00 - 16:30 GMT",
        "index": "^FTSE",
        "index_name": "FTSE 100",
    },
}


# ============================================================================
# PREDEFINED STOCK LISTS PER EXCHANGE
# ============================================================================

STOCK_LISTS = {
    "NSE": {
        "NIFTY50": [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
            "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "WIPRO", "HCLTECH",
            "JSWSTEEL", "BAJAJFINSV", "TITAN", "ULTRACEMCO", "NESTLEIND",
            "DRREDDY", "ADANIPORTS", "TATAMOTORS", "TATASTEEL", "NTPC",
            "POWERGRID", "M&M", "COALINDIA", "ONGC", "IOC", "BPCL",
            "DIVISLAB", "EICHERMOT", "HEROMOTOCO", "SHREECEM", "BRITANNIA",
            "CIPLA", "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO", "TECHM",
            "TATACONSUM", "GRASIM", "HINDALCO", "INDUSINDBK", "UPL",
        ],
        "BANKNIFTY": [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB",
            "BANDHANBNK", "AUBANK",
        ],
        "NIFTY_IT": [
            "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
            "MPHASIS", "COFORGE", "PERSISTENT", "LTTS",
        ],
        "NIFTY_PHARMA": [
            "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
            "AUROPHARMA", "BIOCON", "TORNTPHARM", "ALKEM", "IPCALAB",
        ],
    },
    "BSE": {
        "SENSEX30": [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
            "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "WIPRO", "HCLTECH",
            "TITAN", "ULTRACEMCO", "NESTLEIND", "TATAMOTORS", "TATASTEEL",
            "NTPC", "POWERGRID", "M&M", "TECHM", "INDUSINDBK", "BAJAJ-AUTO",
            "JSWSTEEL",
        ],
    },
    "NYSE": {
        "DOW30": [
            "AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "UNH", "PG", "HD",
            "DIS", "MA", "NVDA", "PYPL", "BAC", "INTC", "CMCSA", "NFLX",
            "KO", "PEP", "T", "MRK", "VZ", "CSCO", "ABT", "XOM", "CVX",
            "PFE", "TMO", "AVGO", "ACN",
        ],
        "MEGA_CAP": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "BRK-B", "UNH", "JNJ", "V", "JPM", "WMT", "MA", "PG",
        ],
        "FINANCIALS": [
            "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW",
            "AXP", "USB", "PNC", "TFC", "COF", "BK",
        ],
    },
    "NASDAQ": {
        "NASDAQ100": [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA",
            "AVGO", "COST", "PEP", "CSCO", "ADBE", "NFLX", "CMCSA", "AMD",
            "INTC", "INTU", "QCOM", "TXN", "AMGN", "ISRG", "BKNG", "AMAT",
            "HON", "SBUX", "VRTX", "GILD", "MDLZ", "ADI",
        ],
        "TECH_GROWTH": [
            "AAPL", "MSFT", "NVDA", "META", "TSLA", "GOOGL", "AMZN",
            "CRM", "ORCL", "ADBE", "NOW", "UBER", "ABNB", "SQ", "SHOP",
        ],
    },
    "LSE": {
        "FTSE100": [
            "SHEL", "AZN", "HSBA", "ULVR", "BP", "GSK", "RIO", "DGE",
            "BHP", "BATS", "REL", "LSEG", "NG", "VOD", "PRU",
            "ABF", "AAL", "ANTO", "AHT", "BARC",
        ],
    },
}


# ============================================================================
# MULTI-EXCHANGE DATA HANDLER
# ============================================================================

class MultiExchangeHandler:
    """Handle data from multiple stock exchanges"""

    def __init__(self):
        self.exchange_config = EXCHANGE_CONFIG
        self.stock_lists = STOCK_LISTS
        self._cache = {}
        self._cache_lock = threading.Lock()
        self.realtime_mode = True
        self.cache_ttl_seconds = 300
        self.realtime_cache_ttl_seconds = 8
        self.max_parallel_workers = 8
        self.max_bulk_chunk_size = 180
        self._latency_ema = None
        self._latency_ema_alpha = 0.25
        self._last_bulk_meta = {}

    def set_realtime_mode(self, enabled=True):
        """Enable or disable realtime mode for market data retrieval."""
        previous_mode = self.realtime_mode
        self.realtime_mode = bool(enabled)
        if self.realtime_mode and not previous_mode:
            self.clear_cache()

    def clear_cache(self):
        """Clear in-memory OHLCV cache."""
        with self._cache_lock:
            self._cache.clear()

    def _get_effective_cache_ttl(self):
        """Use short TTL in realtime mode and longer TTL otherwise."""
        return self.realtime_cache_ttl_seconds if self.realtime_mode else self.cache_ttl_seconds

    def _update_latency_ema(self, sample_seconds):
        """Track moving average of per-symbol fetch latency for adaptive tuning."""
        try:
            sample = float(sample_seconds)
        except Exception:
            return

        if sample <= 0:
            return

        if self._latency_ema is None:
            self._latency_ema = sample
        else:
            self._latency_ema = (
                self._latency_ema_alpha * sample
                + (1 - self._latency_ema_alpha) * self._latency_ema
            )

    def get_adaptive_worker_count(self, task_size, exchange=None, interval="1d", user_cap=None):
        """Compute an adaptive worker count to reduce latency and avoid API throttling."""
        try:
            size = max(1, int(task_size))
        except Exception:
            size = 1

        cpu_count = os.cpu_count() or 4
        cpu_cap = max(4, min(24, cpu_count * 2))
        exchange_cap = {
            "NSE": 14,
            "BSE": 12,
            "NYSE": 18,
            "NASDAQ": 18,
            "LSE": 12,
        }.get(exchange, cpu_cap)

        cap = min(cpu_cap, exchange_cap)

        if str(interval).lower().endswith("m"):
            cap = min(cap, 8)

        if size <= 20:
            cap = min(cap, 6)
        elif size <= 60:
            cap = min(cap, 10)

        if self._latency_ema is not None:
            if self._latency_ema > 2.5:
                cap = min(cap, 4)
            elif self._latency_ema > 1.2:
                cap = min(cap, 6)

        if user_cap is not None:
            try:
                cap = min(cap, max(1, int(user_cap)))
            except Exception:
                pass

        cap = min(cap, self.max_parallel_workers)
        return max(1, min(cap, size))

    def get_last_bulk_meta(self):
        """Return metadata from the last bulk fetch operation."""
        return dict(self._last_bulk_meta)

    def _minimum_rows_required(self, period, interval):
        """Allow short lookbacks for alerting while preserving longer analyses."""
        short_periods = {"1d", "5d", "7d", "1wk", "1mo"}
        if str(interval).lower().endswith("m"):
            return 2
        if str(period).lower() in short_periods:
            return 2
        return 30

    def _inject_latest_quote(self, df, ticker):
        """Patch latest quote into final row so users see fresher realtime values."""
        if df is None or df.empty:
            return df

        live_price = None
        live_volume = None

        try:
            fast_info = getattr(ticker, "fast_info", {}) or {}
            live_price = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
            live_volume = fast_info.get("lastVolume") or fast_info.get("regularMarketVolume")
        except Exception:
            pass

        if live_price is None:
            try:
                info = ticker.info or {}
                live_price = info.get("regularMarketPrice") or info.get("currentPrice")
                live_volume = live_volume or info.get("regularMarketVolume")
            except Exception:
                pass

        if live_price is None:
            return df

        updated = df.copy()
        last_idx = updated.index[-1]

        try:
            live_price = float(live_price)
        except Exception:
            return updated

        if "Close" in updated.columns:
            updated.at[last_idx, "Close"] = live_price

        if "High" in updated.columns:
            try:
                updated.at[last_idx, "High"] = max(float(updated.at[last_idx, "High"]), live_price)
            except Exception:
                updated.at[last_idx, "High"] = live_price

        if "Low" in updated.columns:
            try:
                updated.at[last_idx, "Low"] = min(float(updated.at[last_idx, "Low"]), live_price)
            except Exception:
                updated.at[last_idx, "Low"] = live_price

        if "Volume" in updated.columns and live_volume is not None:
            try:
                updated.at[last_idx, "Volume"] = max(float(updated.at[last_idx, "Volume"]), float(live_volume))
            except Exception:
                pass

        return updated

    def get_supported_exchanges(self):
        """Get list of supported exchanges"""
        return list(self.exchange_config.keys())

    def get_exchange_info(self, exchange):
        """Get exchange configuration"""
        return self.exchange_config.get(exchange, {})

    def get_stock_lists(self, exchange):
        """Get available stock lists for an exchange"""
        return self.stock_lists.get(exchange, {})

    def get_symbol_with_suffix(self, symbol, exchange):
        """Add exchange suffix to symbol"""
        suffix = self.exchange_config.get(exchange, {}).get("suffix", "")
        if suffix and not symbol.endswith(suffix):
            return symbol + suffix
        return symbol

    def get_stock_data(self, symbol, exchange, period="2y", interval="1d", use_cache=None, include_live_quote=True):
        """Fetch stock data for any exchange with realtime-first defaults."""
        if use_cache is None:
            use_cache = True

        cache_key = f"{symbol}_{exchange}_{period}_{interval}"
        effective_ttl = self._get_effective_cache_ttl()
        if use_cache:
            with self._cache_lock:
                if cache_key in self._cache:
                    cached_time, cached_data = self._cache[cache_key]
                    if (datetime.now() - cached_time).total_seconds() < effective_ttl:
                        return cached_data.copy()

        try:
            full_symbol = self.get_symbol_with_suffix(symbol, exchange)
            ticker = yf.Ticker(full_symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False, prepost=True)

            min_rows = self._minimum_rows_required(period, interval)
            if df.empty or len(df) < min_rows:
                return None

            if include_live_quote:
                df = self._inject_latest_quote(df, ticker)

            if use_cache:
                with self._cache_lock:
                    self._cache[cache_key] = (datetime.now(), df.copy())

            return df
        except Exception as e:
            print(f"Error fetching {symbol} from {exchange}: {e}")
            return None

    def get_bulk_stock_data(self, symbols, exchange, period="2y", interval="1d", max_workers=None, include_live_quote=True):
        """Fetch multiple symbols concurrently while preserving realtime behavior."""
        if not symbols:
            return {}

        unique_symbols = list(dict.fromkeys([str(s).strip().upper() for s in symbols if s]))
        if not unique_symbols:
            return {}

        if max_workers is None:
            max_workers = self.max_parallel_workers

        results = {}
        total_requested = len(unique_symbols)
        workers_used = []
        chunk_count = 0

        chunk_size = max(1, min(self.max_bulk_chunk_size, total_requested))

        def _worker(sym):
            data = self.get_stock_data(
                sym,
                exchange,
                period=period,
                interval=interval,
                use_cache=True,
                include_live_quote=include_live_quote,
            )
            return sym, data

        for start in range(0, total_requested, chunk_size):
            chunk = unique_symbols[start:start + chunk_size]
            adaptive_workers = self.get_adaptive_worker_count(
                len(chunk),
                exchange=exchange,
                interval=interval,
                user_cap=max_workers,
            )
            workers_used.append(adaptive_workers)
            chunk_count += 1

            chunk_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=adaptive_workers) as executor:
                futures = [executor.submit(_worker, sym) for sym in chunk]
                for future in as_completed(futures):
                    try:
                        sym, data = future.result()
                        if data is not None:
                            results[sym] = data
                    except Exception:
                        continue

            chunk_elapsed = time.perf_counter() - chunk_start
            per_symbol_latency = chunk_elapsed / max(1, len(chunk))
            self._update_latency_ema(per_symbol_latency)

        if workers_used:
            self._last_bulk_meta = {
                "requested": total_requested,
                "fetched": len(results),
                "chunks": chunk_count,
                "workers_min": min(workers_used),
                "workers_max": max(workers_used),
                "latency_ema": self._latency_ema,
                "realtime_mode": self.realtime_mode,
                "cache_ttl_seconds": self._get_effective_cache_ttl(),
            }

        return results

    def get_stock_info(self, symbol, exchange):
        """Get stock info (fundamentals) for any exchange"""
        try:
            full_symbol = self.get_symbol_with_suffix(symbol, exchange)
            ticker = yf.Ticker(full_symbol)
            info = ticker.info
            return info
        except:
            return {}

    def get_index_data(self, exchange, period="1y"):
        """Get market index data for an exchange"""
        try:
            index_symbol = self.exchange_config[exchange]["index"]
            ticker = yf.Ticker(index_symbol)
            df = ticker.history(period=period)
            return df if not df.empty else None
        except:
            return None

    def get_multiple_exchanges_data(self, symbol_exchange_pairs, period="2y"):
        """Fetch data for multiple symbols across exchanges"""
        results = {}
        for symbol, exchange in symbol_exchange_pairs:
            data = self.get_stock_data(symbol, exchange, period)
            if data is not None:
                results[(symbol, exchange)] = data
        return results

    def load_exchange_stocks(self, exchange):
        """Load all available stocks for an exchange"""
        if exchange == "NSE":
            return self._load_nse_stocks()
        elif exchange == "BSE":
            return self._load_bse_stocks()
        elif exchange in ("NYSE", "NASDAQ"):
            return self._load_us_stocks(exchange)
        elif exchange == "LSE":
            return self._load_lse_stocks()
        return []

    def _load_nse_stocks(self):
        """Load NSE stock list"""
        try:
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            df = pd.read_csv(StringIO(response.text))
            return sorted(df["SYMBOL"].str.strip().tolist())
        except:
            # Fallback to NIFTY50
            return self.stock_lists.get("NSE", {}).get("NIFTY50", [])

    def _load_bse_stocks(self):
        """Load BSE stocks (fallback to predefined)"""
        return self.stock_lists.get("BSE", {}).get("SENSEX30", [])

    def _load_us_stocks(self, exchange):
        """Load US stocks"""
        if exchange == "NYSE":
            lists = self.stock_lists.get("NYSE", {})
        else:
            lists = self.stock_lists.get("NASDAQ", {})

        all_stocks = set()
        for group_stocks in lists.values():
            all_stocks.update(group_stocks)
        return sorted(list(all_stocks))

    def _load_lse_stocks(self):
        """Load LSE stocks"""
        return self.stock_lists.get("LSE", {}).get("FTSE100", [])

    def get_currency_symbol(self, exchange):
        """Get currency symbol for exchange"""
        return self.exchange_config.get(exchange, {}).get("currency_symbol", "$")

    def format_price(self, price, exchange):
        """Format price with currency symbol"""
        symbol = self.get_currency_symbol(exchange)
        return f"{symbol}{price:,.2f}"


# ============================================================================
# MARKET OVERVIEW
# ============================================================================

class MarketOverview:
    """Market overview with multi-exchange indices"""

    def __init__(self, exchange_handler: MultiExchangeHandler):
        self.handler = exchange_handler

    def _fetch_index_snapshot(self, exchange, config):
        """Fetch a single exchange index snapshot."""
        try:
            ticker = yf.Ticker(config["index"])
            hist = ticker.history(period="5d", auto_adjust=False, prepost=True)
            if hist.empty or len(hist) < 2:
                return exchange, None

            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change = current - prev
            change_pct = (change / prev) * 100 if prev else 0.0

            return exchange, {
                "name": config["index_name"],
                "value": current,
                "change": change,
                "change_pct": change_pct,
                "currency": config["currency_symbol"],
            }
        except Exception:
            return exchange, None

    def get_all_indices(self):
        """Get current data for all market indices"""
        indices = {}

        workers = min(max(1, len(EXCHANGE_CONFIG)), 6)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._fetch_index_snapshot, exchange, config)
                for exchange, config in EXCHANGE_CONFIG.items()
            ]

            for future in as_completed(futures):
                try:
                    exchange, payload = future.result()
                    if payload is not None:
                        indices[exchange] = payload
                except Exception:
                    continue

        return indices

    def get_sector_performance(self, exchange="NSE"):
        """Get sector performance for an exchange"""
        sector_mapping = {
            "NSE": {
                "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
                "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
                "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN"],
                "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
                "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
                "Energy": ["RELIANCE", "ONGC", "IOC", "BPCL", "NTPC"],
                "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
                "Infra": ["LT", "ADANIPORTS", "GRASIM", "ULTRACEMCO"],
            },
            "NYSE": {
                "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ACN"],
                "Healthcare": ["JNJ", "UNH", "PFE", "ABT", "TMO"],
                "Financials": ["JPM", "BAC", "GS", "V", "MA"],
                "Energy": ["XOM", "CVX"],
                "Consumer": ["WMT", "PG", "KO", "PEP", "HD"],
            },
        }

        sectors = sector_mapping.get(exchange, {})
        sector_data = {}

        for sector, symbols in sectors.items():
            returns = []
            for symbol in symbols:
                try:
                    df = self.handler.get_stock_data(symbol, exchange, period="1mo")
                    if df is not None and len(df) > 1:
                        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
                        returns.append(ret)
                except:
                    pass

            if returns:
                sector_data[sector] = {
                    "avg_return": np.mean(returns),
                    "best": max(returns),
                    "worst": min(returns),
                    "stocks_count": len(returns),
                }

        return sector_data

    def get_market_breadth(self, exchange="NSE"):
        """Compute market breadth (advance/decline)"""
        stock_list_key = next(iter(self.handler.get_stock_lists(exchange)), None)
        if not stock_list_key:
            return None

        stocks = self.handler.stock_lists[exchange][stock_list_key]
        advancing = 0
        declining = 0
        unchanged = 0

        for symbol in stocks:
            try:
                df = self.handler.get_stock_data(symbol, exchange, period="5d")
                if df is not None and len(df) >= 2:
                    change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
                    if change > 0:
                        advancing += 1
                    elif change < 0:
                        declining += 1
                    else:
                        unchanged += 1
            except:
                pass

        total = advancing + declining + unchanged
        return {
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "total": total,
            "breadth_ratio": advancing / declining if declining > 0 else float("inf"),
        }


# ============================================================================
# CROSS-EXCHANGE COMPARISON
# ============================================================================

class CrossExchangeComparator:
    """Compare stocks across different exchanges"""

    def __init__(self, exchange_handler: MultiExchangeHandler):
        self.handler = exchange_handler

    def compare_stocks(self, stock_exchange_pairs, period="1y"):
        """Compare performance of stocks across exchanges"""
        comparison = {}
        for symbol, exchange in stock_exchange_pairs:
            df = self.handler.get_stock_data(symbol, exchange, period)
            if df is None:
                continue

            current_price = df["Close"].iloc[-1]
            start_price = df["Close"].iloc[0]
            returns = (current_price / start_price - 1) * 100
            volatility = df["Close"].pct_change().std() * np.sqrt(252) * 100
            max_dd = self._calc_max_drawdown(df["Close"])
            sharpe = self._calc_sharpe(df["Close"])
            avg_volume = df["Volume"].mean()

            comparison[(symbol, exchange)] = {
                "current_price": current_price,
                "period_return": returns,
                "volatility": volatility,
                "max_drawdown": max_dd,
                "sharpe_ratio": sharpe,
                "avg_volume": avg_volume,
                "currency": self.handler.get_currency_symbol(exchange),
            }

        return comparison

    def get_correlation_matrix(self, stock_exchange_pairs, period="1y"):
        """Compute correlation matrix across exchanges"""
        returns_data = {}
        for symbol, exchange in stock_exchange_pairs:
            df = self.handler.get_stock_data(symbol, exchange, period)
            if df is not None:
                returns_data[f"{symbol} ({exchange})"] = df["Close"].pct_change().dropna()

        if len(returns_data) < 2:
            return None

        # Align dates
        combined = pd.DataFrame(returns_data)
        combined = combined.dropna()
        return combined.corr()

    def _calc_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min() * 100

    def _calc_sharpe(self, prices, risk_free_rate=0.05):
        """Calculate Sharpe ratio"""
        returns = prices.pct_change().dropna()
        excess = returns.mean() * 252 - risk_free_rate
        std = returns.std() * np.sqrt(252)
        return excess / std if std > 0 else 0
