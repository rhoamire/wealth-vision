"""
Technical Chart Pattern Detection Module
Detects support/resistance, chart patterns, candlestick patterns, and divergences
"""

import pandas as pd
import numpy as np
import ta
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# SUPPORT & RESISTANCE DETECTOR
# ============================================================================

class SupportResistanceDetector:
    """Detect key support and resistance levels"""

    def __init__(self, lookback=120, num_levels=5, tolerance_pct=1.5):
        self.lookback = lookback
        self.num_levels = num_levels
        self.tolerance_pct = tolerance_pct

    def detect_levels(self, df):
        """Detect support and resistance levels using multiple methods"""
        if df is None or len(df) < 50:
            return {"support": [], "resistance": [], "pivot_points": {}}

        close = df["Close"].values if isinstance(df["Close"], pd.Series) else df["Close"].iloc[:, 0].values
        high = df["High"].values if isinstance(df["High"], pd.Series) else df["High"].iloc[:, 0].values
        low = df["Low"].values if isinstance(df["Low"], pd.Series) else df["Low"].iloc[:, 0].values

        current_price = close[-1]

        # Method 1: Local extrema
        extrema_levels = self._local_extrema_levels(close, high, low)

        # Method 2: Volume profile levels
        volume_levels = self._volume_profile_levels(df)

        # Method 3: Pivot points
        pivot_data = self._pivot_points(high, low, close)

        # Method 4: Fibonacci retracement
        fib_levels = self._fibonacci_levels(high, low)

        # Combine all levels
        all_levels = extrema_levels + volume_levels + list(fib_levels.values())

        # Cluster nearby levels
        support, resistance = self._cluster_levels(all_levels, current_price)

        return {
            "support": support[:self.num_levels],
            "resistance": resistance[:self.num_levels],
            "pivot_points": pivot_data,
            "fibonacci": fib_levels,
            "current_price": current_price,
            "nearest_support": support[0] if support else None,
            "nearest_resistance": resistance[0] if resistance else None,
        }

    def _local_extrema_levels(self, close, high, low, order=10):
        """Find local minima/maxima as S/R levels"""
        levels = []
        try:
            local_max_idx = argrelextrema(high[-self.lookback:], np.greater, order=order)[0]
            local_min_idx = argrelextrema(low[-self.lookback:], np.less, order=order)[0]

            for idx in local_max_idx:
                levels.append(high[-self.lookback:][idx])
            for idx in local_min_idx:
                levels.append(low[-self.lookback:][idx])
        except Exception:
            pass
        return levels

    def _volume_profile_levels(self, df, bins=20):
        """Find high-volume price levels"""
        levels = []
        try:
            data = df.tail(self.lookback)
            close = data["Close"].values if isinstance(data["Close"], pd.Series) else data["Close"].iloc[:, 0].values
            volume = data["Volume"].values if isinstance(data["Volume"], pd.Series) else data["Volume"].iloc[:, 0].values

            price_range = np.linspace(close.min(), close.max(), bins + 1)
            for i in range(bins):
                mask = (close >= price_range[i]) & (close < price_range[i + 1])
                if mask.any():
                    vol_at_level = volume[mask].sum()
                    price_at_level = (price_range[i] + price_range[i + 1]) / 2
                    levels.append((price_at_level, vol_at_level))

            # Top volume levels
            levels.sort(key=lambda x: x[1], reverse=True)
            levels = [l[0] for l in levels[:5]]
        except Exception:
            pass
        return levels

    def _pivot_points(self, high, low, close):
        """Calculate standard pivot points"""
        h = high[-1]
        l = low[-1]
        c = close[-1]
        pivot = (h + l + c) / 3

        return {
            "pivot": pivot,
            "r1": 2 * pivot - l,
            "r2": pivot + (h - l),
            "r3": h + 2 * (pivot - l),
            "s1": 2 * pivot - h,
            "s2": pivot - (h - l),
            "s3": l - 2 * (h - pivot),
        }

    def _fibonacci_levels(self, high, low):
        """Fibonacci retracement levels"""
        h = np.max(high[-self.lookback:])
        l = np.min(low[-self.lookback:])
        diff = h - l

        return {
            "fib_0": l,
            "fib_236": l + 0.236 * diff,
            "fib_382": l + 0.382 * diff,
            "fib_500": l + 0.5 * diff,
            "fib_618": l + 0.618 * diff,
            "fib_786": l + 0.786 * diff,
            "fib_100": h,
        }

    def _cluster_levels(self, all_levels, current_price):
        """Cluster nearby levels and separate into support/resistance"""
        if not all_levels:
            return [], []

        tolerance = current_price * (self.tolerance_pct / 100)
        levels = sorted(set(all_levels))

        # Cluster nearby levels
        clustered = []
        current_cluster = [levels[0]]
        for i in range(1, len(levels)):
            if levels[i] - levels[i - 1] < tolerance:
                current_cluster.append(levels[i])
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [levels[i]]
        clustered.append(np.mean(current_cluster))

        # Separate into support (below price) and resistance (above price)
        support = sorted([l for l in clustered if l < current_price], reverse=True)
        resistance = sorted([l for l in clustered if l >= current_price])

        return support, resistance


# ============================================================================
# CHART PATTERN DETECTOR
# ============================================================================

class ChartPatternDetector:
    """Detect common chart patterns"""

    def detect_all_patterns(self, df):
        """Run all pattern detection on stock data"""
        if df is None or len(df) < 60:
            return {"patterns": [], "candlestick_patterns": []}

        patterns = []

        # Trend patterns
        trend = self._detect_trend(df)
        if trend:
            patterns.append(trend)

        # Double top/bottom
        double_pattern = self._detect_double_top_bottom(df)
        if double_pattern:
            patterns.append(double_pattern)

        # Head and shoulders
        hs = self._detect_head_shoulders(df)
        if hs:
            patterns.append(hs)

        # Triangle patterns
        triangle = self._detect_triangle(df)
        if triangle:
            patterns.append(triangle)

        # Bollinger squeeze
        squeeze = self._detect_bollinger_squeeze(df)
        if squeeze:
            patterns.append(squeeze)

        # Moving average convergence
        ma_conv = self._detect_ma_convergence(df)
        if ma_conv:
            patterns.append(ma_conv)

        # Volume patterns
        vol = self._detect_volume_breakout(df)
        if vol:
            patterns.append(vol)

        # Gap detection
        gap = self._detect_gap(df)
        if gap:
            patterns.append(gap)

        # Candlestick patterns
        candle_patterns = self._detect_candlestick_patterns(df)

        return {
            "patterns": patterns,
            "candlestick_patterns": candle_patterns,
            "pattern_count": len(patterns) + len(candle_patterns),
        }

    def _detect_trend(self, df, window=50):
        """Detect current trend and strength"""
        try:
            close = df["Close"].tail(window)
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()

            adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
            adx_val = adx.adx().iloc[-1]

            if close.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend = "Uptrend"
            elif close.iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1]:
                trend = "Downtrend"
            else:
                trend = "Sideways"

            strength = "Strong" if adx_val > 25 else "Weak"

            return {
                "pattern": f"{strength} {trend}",
                "type": "trend",
                "signal": "Bullish" if trend == "Uptrend" else ("Bearish" if trend == "Downtrend" else "Neutral"),
                "confidence": min(adx_val * 2, 100),
                "detail": f"ADX: {adx_val:.1f}, Price {'above' if trend == 'Uptrend' else 'below'} key SMAs",
            }
        except Exception:
            return None

    def _detect_double_top_bottom(self, df, lookback=60):
        """Detect double top or double bottom"""
        try:
            close = df["Close"].values[-lookback:]
            high = df["High"].values[-lookback:]
            low = df["Low"].values[-lookback:]

            # Find peaks and troughs
            peaks = argrelextrema(high, np.greater, order=5)[0]
            troughs = argrelextrema(low, np.less, order=5)[0]

            # Double top: two peaks at similar levels with valley between
            if len(peaks) >= 2:
                p1, p2 = high[peaks[-2]], high[peaks[-1]]
                tolerance = p1 * 0.02
                if abs(p1 - p2) < tolerance and peaks[-1] - peaks[-2] > 10:
                    return {
                        "pattern": "Double Top",
                        "type": "reversal",
                        "signal": "Bearish",
                        "confidence": 70,
                        "detail": f"Two peaks at ~{p1:.2f} and ~{p2:.2f}",
                    }

            # Double bottom
            if len(troughs) >= 2:
                t1, t2 = low[troughs[-2]], low[troughs[-1]]
                tolerance = t1 * 0.02
                if abs(t1 - t2) < tolerance and troughs[-1] - troughs[-2] > 10:
                    return {
                        "pattern": "Double Bottom",
                        "type": "reversal",
                        "signal": "Bullish",
                        "confidence": 70,
                        "detail": f"Two troughs at ~{t1:.2f} and ~{t2:.2f}",
                    }
        except Exception:
            pass
        return None

    def _detect_head_shoulders(self, df, lookback=80):
        """Detect Head and Shoulders or Inverse H&S"""
        try:
            high = df["High"].values[-lookback:]
            low = df["Low"].values[-lookback:]

            peaks = argrelextrema(high, np.greater, order=7)[0]
            troughs = argrelextrema(low, np.less, order=7)[0]

            # Need at least 3 peaks for H&S
            if len(peaks) >= 3:
                left = high[peaks[-3]]
                head = high[peaks[-2]]
                right = high[peaks[-1]]

                if head > left and head > right and abs(left - right) / left < 0.05:
                    return {
                        "pattern": "Head and Shoulders",
                        "type": "reversal",
                        "signal": "Bearish",
                        "confidence": 75,
                        "detail": f"Left: {left:.2f}, Head: {head:.2f}, Right: {right:.2f}",
                    }

            # Inverse H&S
            if len(troughs) >= 3:
                left = low[troughs[-3]]
                head = low[troughs[-2]]
                right = low[troughs[-1]]

                if head < left and head < right and abs(left - right) / left < 0.05:
                    return {
                        "pattern": "Inverse Head and Shoulders",
                        "type": "reversal",
                        "signal": "Bullish",
                        "confidence": 75,
                        "detail": f"Left: {left:.2f}, Head: {head:.2f}, Right: {right:.2f}",
                    }
        except Exception:
            pass
        return None

    def _detect_triangle(self, df, lookback=40):
        """Detect ascending, descending, and symmetrical triangles"""
        try:
            high = df["High"].values[-lookback:]
            low = df["Low"].values[-lookback:]

            # Linear regression on highs and lows
            x = np.arange(lookback)

            high_slope = np.polyfit(x, high, 1)[0]
            low_slope = np.polyfit(x, low, 1)[0]

            range_start = high[0] - low[0]
            range_end = high[-1] - low[-1]

            if range_end < range_start * 0.6:  # Converging
                if high_slope < 0 and low_slope > 0:
                    pattern = "Symmetrical Triangle"
                    signal = "Neutral"
                    conf = 65
                elif abs(high_slope) < abs(low_slope) * 0.3 and low_slope > 0:
                    pattern = "Ascending Triangle"
                    signal = "Bullish"
                    conf = 70
                elif abs(low_slope) < abs(high_slope) * 0.3 and high_slope < 0:
                    pattern = "Descending Triangle"
                    signal = "Bearish"
                    conf = 70
                else:
                    return None

                return {
                    "pattern": pattern,
                    "type": "continuation",
                    "signal": signal,
                    "confidence": conf,
                    "detail": f"Range narrowing from {range_start:.2f} to {range_end:.2f}",
                }
        except Exception:
            pass
        return None

    def _detect_bollinger_squeeze(self, df):
        """Detect Bollinger Band squeeze (low volatility compression)"""
        try:
            bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
            bandwidth = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

            current_bw = bandwidth.iloc[-1]
            avg_bw = bandwidth.rolling(120).mean().iloc[-1]

            if current_bw < avg_bw * 0.6:
                return {
                    "pattern": "Bollinger Squeeze",
                    "type": "volatility",
                    "signal": "Expansion Expected",
                    "confidence": 70,
                    "detail": f"Bandwidth: {current_bw:.4f} vs avg: {avg_bw:.4f} (compressed {(1-current_bw/avg_bw)*100:.0f}%)",
                }
        except Exception:
            pass
        return None

    def _detect_ma_convergence(self, df):
        """Detect when multiple MAs are converging (breakout setup)"""
        try:
            close = df["Close"]
            sma_10 = close.rolling(10).mean().iloc[-1]
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]

            avg_price = (sma_10 + sma_20 + sma_50) / 3
            spread = max(sma_10, sma_20, sma_50) - min(sma_10, sma_20, sma_50)
            spread_pct = spread / avg_price * 100

            if spread_pct < 1.5:
                return {
                    "pattern": "MA Convergence",
                    "type": "breakout_setup",
                    "signal": "Breakout Imminent",
                    "confidence": 65,
                    "detail": f"SMA10={sma_10:.2f}, SMA20={sma_20:.2f}, SMA50={sma_50:.2f} (spread: {spread_pct:.2f}%)",
                }
        except Exception:
            pass
        return None

    def _detect_volume_breakout(self, df, lookback=5):
        """Detect unusual volume activity"""
        try:
            recent_vol = df["Volume"].tail(lookback).mean()
            avg_vol = df["Volume"].rolling(50).mean().iloc[-1]
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

            if vol_ratio > 2.0:
                price_change = (df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1) * 100
                direction = "Bullish" if price_change > 0 else "Bearish"
                return {
                    "pattern": "Volume Breakout",
                    "type": "volume",
                    "signal": direction,
                    "confidence": min(vol_ratio * 25, 90),
                    "detail": f"Volume {vol_ratio:.1f}x average, price change: {price_change:.2f}%",
                }
        except Exception:
            pass
        return None

    def _detect_gap(self, df):
        """Detect price gaps"""
        try:
            prev_close = df["Close"].iloc[-2]
            curr_open = df["Open"].iloc[-1]
            curr_close = df["Close"].iloc[-1]
            gap_pct = (curr_open / prev_close - 1) * 100

            if abs(gap_pct) > 2.0:
                direction = "Gap Up" if gap_pct > 0 else "Gap Down"
                filled = "Filled" if (gap_pct > 0 and curr_close < prev_close) or \
                                     (gap_pct < 0 and curr_close > prev_close) else "Unfilled"
                return {
                    "pattern": f"{direction} ({filled})",
                    "type": "gap",
                    "signal": "Bullish" if gap_pct > 0 else "Bearish",
                    "confidence": min(abs(gap_pct) * 10, 85),
                    "detail": f"Gap: {gap_pct:.2f}%, Previous close: {prev_close:.2f}, Open: {curr_open:.2f}",
                }
        except Exception:
            pass
        return None

    def _detect_candlestick_patterns(self, df):
        """Detect common candlestick patterns"""
        patterns = []
        try:
            o, h, l, c = df["Open"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
            o2, h2, l2, c2 = df["Open"].iloc[-2], df["High"].iloc[-2], df["Low"].iloc[-2], df["Close"].iloc[-2]

            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            full_range = h - l

            if full_range == 0:
                return patterns

            body_pct = body / full_range

            # Doji
            if body_pct < 0.1:
                patterns.append({
                    "pattern": "Doji", "signal": "Reversal/Indecision",
                    "confidence": 55, "detail": "Body < 10% of range"
                })

            # Hammer / Hanging Man
            if lower_shadow > body * 2 and upper_shadow < body * 0.5 and body_pct > 0.1:
                prev_trend = "down" if c2 < df["Close"].iloc[-5] else "up"
                if prev_trend == "down":
                    patterns.append({
                        "pattern": "Hammer", "signal": "Bullish Reversal",
                        "confidence": 65, "detail": "Long lower shadow after downtrend"
                    })
                else:
                    patterns.append({
                        "pattern": "Hanging Man", "signal": "Bearish Reversal",
                        "confidence": 60, "detail": "Long lower shadow after uptrend"
                    })

            # Shooting Star / Inverted Hammer
            if upper_shadow > body * 2 and lower_shadow < body * 0.5 and body_pct > 0.1:
                prev_trend = "up" if c2 > df["Close"].iloc[-5] else "down"
                if prev_trend == "up":
                    patterns.append({
                        "pattern": "Shooting Star", "signal": "Bearish Reversal",
                        "confidence": 65, "detail": "Long upper shadow after uptrend"
                    })
                else:
                    patterns.append({
                        "pattern": "Inverted Hammer", "signal": "Bullish Reversal",
                        "confidence": 60, "detail": "Long upper shadow after downtrend"
                    })

            # Bullish/Bearish Engulfing
            body2 = abs(c2 - o2)
            if body > body2 and c > o and c2 < o2 and o <= c2 and c >= o2:
                patterns.append({
                    "pattern": "Bullish Engulfing", "signal": "Bullish Reversal",
                    "confidence": 70, "detail": "Current candle engulfs previous bearish candle"
                })
            elif body > body2 and c < o and c2 > o2 and o >= c2 and c <= o2:
                patterns.append({
                    "pattern": "Bearish Engulfing", "signal": "Bearish Reversal",
                    "confidence": 70, "detail": "Current candle engulfs previous bullish candle"
                })

            # Marubozu (strong conviction candle)
            if body_pct > 0.85:
                direction = "Bullish" if c > o else "Bearish"
                patterns.append({
                    "pattern": f"{direction} Marubozu", "signal": f"{direction}",
                    "confidence": 65, "detail": f"Strong {direction.lower()} candle with minimal shadows"
                })

            # Morning Star / Evening Star (requires 3 candles)
            if len(df) >= 3:
                o3, c3 = df["Open"].iloc[-3], df["Close"].iloc[-3]
                body3 = abs(c3 - o3)
                if c3 < o3 and body2 < body3 * 0.3 and c > o and c > (o3 + c3) / 2:
                    patterns.append({
                        "pattern": "Morning Star", "signal": "Bullish Reversal",
                        "confidence": 75, "detail": "Three-candle bullish reversal pattern"
                    })
                elif c3 > o3 and body2 < body3 * 0.3 and c < o and c < (o3 + c3) / 2:
                    patterns.append({
                        "pattern": "Evening Star", "signal": "Bearish Reversal",
                        "confidence": 75, "detail": "Three-candle bearish reversal pattern"
                    })
        except Exception:
            pass

        return patterns


# ============================================================================
# DIVERGENCE DETECTOR
# ============================================================================

class DivergenceDetector:
    """Detect RSI and MACD divergences"""

    def detect_divergences(self, df, lookback=30):
        """Detect all types of divergences"""
        if df is None or len(df) < lookback + 20:
            return []

        divergences = []

        # RSI divergence
        rsi_div = self._rsi_divergence(df, lookback)
        if rsi_div:
            divergences.extend(rsi_div)

        # MACD divergence
        macd_div = self._macd_divergence(df, lookback)
        if macd_div:
            divergences.extend(macd_div)

        return divergences

    def _rsi_divergence(self, df, lookback=30):
        """Detect RSI divergences"""
        results = []
        try:
            close = df["Close"].values
            rsi = ta.momentum.RSIIndicator(df["Close"], window=14).rsi().values

            data = close[-lookback:]
            rsi_data = rsi[-lookback:]

            price_peaks = argrelextrema(data, np.greater, order=5)[0]
            price_troughs = argrelextrema(data, np.less, order=5)[0]

            # Bearish divergence: price higher highs, RSI lower highs
            if len(price_peaks) >= 2:
                p1, p2 = price_peaks[-2], price_peaks[-1]
                if data[p2] > data[p1] and rsi_data[p2] < rsi_data[p1]:
                    results.append({
                        "type": "Bearish RSI Divergence",
                        "signal": "Bearish",
                        "confidence": 70,
                        "detail": f"Price making higher highs but RSI making lower highs (RSI: {rsi_data[p1]:.1f} → {rsi_data[p2]:.1f})",
                    })

            # Bullish divergence: price lower lows, RSI higher lows
            if len(price_troughs) >= 2:
                t1, t2 = price_troughs[-2], price_troughs[-1]
                if data[t2] < data[t1] and rsi_data[t2] > rsi_data[t1]:
                    results.append({
                        "type": "Bullish RSI Divergence",
                        "signal": "Bullish",
                        "confidence": 70,
                        "detail": f"Price making lower lows but RSI making higher lows (RSI: {rsi_data[t1]:.1f} → {rsi_data[t2]:.1f})",
                    })
        except Exception:
            pass
        return results

    def _macd_divergence(self, df, lookback=30):
        """Detect MACD divergences"""
        results = []
        try:
            close = df["Close"].values
            macd_ind = ta.trend.MACD(df["Close"])
            macd_line = macd_ind.macd().values

            data = close[-lookback:]
            macd_data = macd_line[-lookback:]

            price_peaks = argrelextrema(data, np.greater, order=5)[0]
            price_troughs = argrelextrema(data, np.less, order=5)[0]

            # Bearish MACD divergence
            if len(price_peaks) >= 2:
                p1, p2 = price_peaks[-2], price_peaks[-1]
                if data[p2] > data[p1] and macd_data[p2] < macd_data[p1]:
                    results.append({
                        "type": "Bearish MACD Divergence",
                        "signal": "Bearish",
                        "confidence": 65,
                        "detail": f"Price making higher highs but MACD making lower highs",
                    })

            # Bullish MACD divergence
            if len(price_troughs) >= 2:
                t1, t2 = price_troughs[-2], price_troughs[-1]
                if data[t2] < data[t1] and macd_data[t2] > macd_data[t1]:
                    results.append({
                        "type": "Bullish MACD Divergence",
                        "signal": "Bullish",
                        "confidence": 65,
                        "detail": f"Price making lower lows but MACD making higher lows",
                    })
        except Exception:
            pass
        return results


# ============================================================================
# COMPREHENSIVE TECHNICAL ANALYSIS
# ============================================================================

class TechnicalAnalyzer:
    """Complete technical analysis combining all detectors"""

    def __init__(self):
        self.sr_detector = SupportResistanceDetector()
        self.pattern_detector = ChartPatternDetector()
        self.divergence_detector = DivergenceDetector()

    def full_analysis(self, df):
        """Run complete technical analysis"""
        if df is None or len(df) < 50:
            return None

        sr_levels = self.sr_detector.detect_levels(df)
        patterns = self.pattern_detector.detect_all_patterns(df)
        divergences = self.divergence_detector.detect_divergences(df)

        # Overall sentiment score (-100 to 100)
        bullish_signals = 0
        bearish_signals = 0

        for p in patterns["patterns"]:
            if p["signal"] == "Bullish":
                bullish_signals += p["confidence"]
            elif p["signal"] == "Bearish":
                bearish_signals += p["confidence"]

        for p in patterns["candlestick_patterns"]:
            if "Bullish" in p["signal"]:
                bullish_signals += p["confidence"]
            elif "Bearish" in p["signal"]:
                bearish_signals += p["confidence"]

        for d in divergences:
            if d["signal"] == "Bullish":
                bullish_signals += d["confidence"]
            elif d["signal"] == "Bearish":
                bearish_signals += d["confidence"]

        total = bullish_signals + bearish_signals
        sentiment = ((bullish_signals - bearish_signals) / total * 100) if total > 0 else 0

        return {
            "support_resistance": sr_levels,
            "chart_patterns": patterns["patterns"],
            "candlestick_patterns": patterns["candlestick_patterns"],
            "divergences": divergences,
            "total_patterns_found": patterns["pattern_count"] + len(divergences),
            "bullish_score": bullish_signals,
            "bearish_score": bearish_signals,
            "sentiment_score": sentiment,
            "sentiment": "Bullish" if sentiment > 15 else ("Bearish" if sentiment < -15 else "Neutral"),
        }
