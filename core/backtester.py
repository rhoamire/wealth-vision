"""
Strategy Backtesting Engine
Test strategies against historical data with performance metrics
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# BACKTESTER
# ============================================================================

class Backtester:
    """Backtest trading strategies on historical data"""

    def __init__(self, initial_capital=100000, commission_pct=0.001, slippage_pct=0.0005):
        """
        Args:
            initial_capital: Starting capital
            commission_pct: Commission per trade (0.1% default)
            slippage_pct: Assumed slippage per trade
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run_backtest(self, df, strategy_func, strategy_name="Custom", **kwargs):
        """
        Run a backtest.
        
        Args:
            df: OHLCV DataFrame
            strategy_func: Callable(df, **kwargs) -> Series of signals (1=buy, -1=sell, 0=hold)
            strategy_name: Name for reporting
        
        Returns:
            dict with performance metrics and trade log
        """
        if df is None or len(df) < 50:
            return None

        df = df.copy()
        signals = strategy_func(df, **kwargs)

        # Initialize tracking
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        daily_returns = []
        prev_equity = capital

        for i in range(len(df)):
            current_price = df["Close"].iloc[i]
            date = df.index[i]
            signal = signals.iloc[i] if i < len(signals) else 0

            # Execute signal
            if signal == 1 and position == 0:
                # Buy
                entry_cost = current_price * (1 + self.slippage_pct)
                shares = int(capital * 0.95 / entry_cost)  # Use 95% of capital
                if shares > 0:
                    commission = shares * entry_cost * self.commission_pct
                    capital -= shares * entry_cost + commission
                    position = shares
                    entry_price = entry_cost
                    trades.append({
                        "date": date,
                        "type": "BUY",
                        "price": entry_cost,
                        "shares": shares,
                        "commission": commission,
                        "capital_after": capital,
                    })

            elif signal == -1 and position > 0:
                # Sell
                exit_price = current_price * (1 - self.slippage_pct)
                commission = position * exit_price * self.commission_pct
                proceeds = position * exit_price - commission
                pnl = (exit_price - entry_price) * position - commission
                capital += proceeds
                trades.append({
                    "date": date,
                    "type": "SELL",
                    "price": exit_price,
                    "shares": position,
                    "commission": commission,
                    "pnl": pnl,
                    "pnl_pct": (exit_price / entry_price - 1) * 100,
                    "capital_after": capital,
                })
                position = 0
                entry_price = 0

            # Track equity
            equity = capital + position * current_price
            daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0
            daily_returns.append(daily_ret)
            equity_curve.append({
                "date": date,
                "equity": equity,
                "position": position,
                "price": current_price,
            })
            prev_equity = equity

        # Close any open position at end
        if position > 0:
            final_price = df["Close"].iloc[-1] * (1 - self.slippage_pct)
            commission = position * final_price * self.commission_pct
            proceeds = position * final_price - commission
            pnl = (final_price - entry_price) * position - commission
            capital += proceeds
            trades.append({
                "date": df.index[-1],
                "type": "SELL (Close)",
                "price": final_price,
                "shares": position,
                "commission": commission,
                "pnl": pnl,
                "pnl_pct": (final_price / entry_price - 1) * 100,
                "capital_after": capital,
            })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        returns_series = pd.Series(daily_returns, index=df.index[:len(daily_returns)])

        metrics = self._calculate_metrics(
            equity_df, trades_df, returns_series, strategy_name
        )
        metrics["equity_curve"] = equity_df
        metrics["trades"] = trades_df
        metrics["daily_returns"] = returns_series

        return metrics

    def _calculate_metrics(self, equity_df, trades_df, returns, strategy_name):
        """Calculate comprehensive backtest metrics"""
        if equity_df.empty:
            return {"strategy": strategy_name, "error": "No data"}

        final_equity = equity_df["equity"].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        trading_days = len(equity_df)
        years = trading_days / 252

        # CAGR
        cagr = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Volatility
        ann_vol = returns.std() * np.sqrt(252) * 100
        daily_vol = returns.std() * 100

        # Sharpe
        excess = returns.mean() * 252 - 0.06
        sharpe = excess / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Sortino
        downside = returns[returns < 0]
        sortino_denom = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
        sortino = excess / sortino_denom

        # Drawdown
        equity_series = equity_df["equity"]
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min() * 100
        current_dd = drawdown.iloc[-1] * 100

        # Trade statistics
        sell_trades = trades_df[trades_df["type"].str.contains("SELL")] if not trades_df.empty else pd.DataFrame()
        n_trades = len(sell_trades)

        if n_trades > 0 and "pnl" in sell_trades.columns:
            winning = sell_trades[sell_trades["pnl"] > 0]
            losing = sell_trades[sell_trades["pnl"] <= 0]
            win_rate = len(winning) / n_trades * 100
            avg_win = winning["pnl_pct"].mean() if len(winning) > 0 else 0
            avg_loss = losing["pnl_pct"].mean() if len(losing) > 0 else 0
            profit_factor = (abs(winning["pnl"].sum()) / abs(losing["pnl"].sum())
                            if len(losing) > 0 and losing["pnl"].sum() != 0 else float("inf"))
            max_win = sell_trades["pnl_pct"].max() if len(sell_trades) > 0 else 0
            max_loss = sell_trades["pnl_pct"].min() if len(sell_trades) > 0 else 0
            avg_trade_pnl = sell_trades["pnl_pct"].mean()
            total_commission = trades_df["commission"].sum() if "commission" in trades_df.columns else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            max_win = max_loss = avg_trade_pnl = total_commission = 0

        return {
            "strategy": strategy_name,
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "current_drawdown": current_dd,
            "total_trades": n_trades,
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "max_win_pct": max_win,
            "max_loss_pct": max_loss,
            "avg_trade_return": avg_trade_pnl,
            "total_commission": total_commission,
            "trading_days": trading_days,
        }

    def compare_strategies(self, df, strategy_funcs):
        """
        Compare multiple strategies.
        strategy_funcs: dict of {name: callable}
        """
        results = {}
        for name, func in strategy_funcs.items():
            result = self.run_backtest(df, func, strategy_name=name)
            if result:
                results[name] = result
        return results


# ============================================================================
# PREDEFINED STRATEGY SIGNAL GENERATORS
# ============================================================================

def macd_crossover_signals(df, fast=12, slow=26, signal=9):
    """Generate MACD crossover buy/sell signals"""
    macd = ta.trend.MACD(df["Close"], window_fast=fast, window_slow=slow, window_sign=signal)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()

    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i - 1] <= signal_line.iloc[i - 1]:
            signals.iloc[i] = 1  # Buy
        elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i - 1] >= signal_line.iloc[i - 1]:
            signals.iloc[i] = -1  # Sell
    return signals


def rsi_strategy_signals(df, period=14, oversold=30, overbought=70):
    """Generate RSI buy/sell signals"""
    rsi = ta.momentum.RSIIndicator(df["Close"], window=period).rsi()
    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if rsi.iloc[i] > oversold and rsi.iloc[i - 1] <= oversold:
            signals.iloc[i] = 1
        elif rsi.iloc[i] < overbought and rsi.iloc[i - 1] >= overbought:
            signals.iloc[i] = -1
    return signals


def sma_crossover_signals(df, fast=20, slow=50):
    """Generate SMA crossover signals"""
    sma_fast = df["Close"].rolling(fast).mean()
    sma_slow = df["Close"].rolling(slow).mean()
    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if sma_fast.iloc[i] > sma_slow.iloc[i] and sma_fast.iloc[i - 1] <= sma_slow.iloc[i - 1]:
            signals.iloc[i] = 1
        elif sma_fast.iloc[i] < sma_slow.iloc[i] and sma_fast.iloc[i - 1] >= sma_slow.iloc[i - 1]:
            signals.iloc[i] = -1
    return signals


def bollinger_band_signals(df, period=20, std=2):
    """Generate Bollinger Band mean-reversion signals"""
    bb = ta.volatility.BollingerBands(df["Close"], window=period, window_dev=std)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if df["Close"].iloc[i] < lower.iloc[i] and df["Close"].iloc[i - 1] >= lower.iloc[i - 1]:
            signals.iloc[i] = 1
        elif df["Close"].iloc[i] > upper.iloc[i] and df["Close"].iloc[i - 1] <= upper.iloc[i - 1]:
            signals.iloc[i] = -1
    return signals


def adx_dmi_signals(df, period=14, adx_threshold=25):
    """Generate ADX/DMI signals"""
    adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=period)
    adx = adx_ind.adx()
    dmi_plus = adx_ind.adx_pos()
    dmi_minus = adx_ind.adx_neg()
    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if (adx.iloc[i] > adx_threshold and
                dmi_plus.iloc[i] > dmi_minus.iloc[i] and
                dmi_plus.iloc[i - 1] <= dmi_minus.iloc[i - 1]):
            signals.iloc[i] = 1
        elif (dmi_plus.iloc[i] < dmi_minus.iloc[i] and
              dmi_plus.iloc[i - 1] >= dmi_minus.iloc[i - 1]):
            signals.iloc[i] = -1
    return signals


def supertrend_signals(df, period=14, multiplier=3):
    """Generate Supertrend buy/sell signals"""
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=period).average_true_range()
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    signals = pd.Series(0, index=df.index)
    in_uptrend = True

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper_band.iloc[i - 1]:
            if not in_uptrend:
                signals.iloc[i] = 1
            in_uptrend = True
        elif df["Close"].iloc[i] < lower_band.iloc[i - 1]:
            if in_uptrend:
                signals.iloc[i] = -1
            in_uptrend = False
    return signals


def combined_strategy_signals(df):
    """Combined multi-indicator strategy"""
    rsi = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    sma_20 = df["Close"].rolling(20).mean()
    sma_50 = df["Close"].rolling(50).mean()
    vol_ma = df["Volume"].rolling(20).mean()

    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        buy_score = 0

        # MACD crossover
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i - 1] <= signal_line.iloc[i - 1]:
            buy_score += 2
        # RSI oversold recovery
        if 40 < rsi.iloc[i] < 65 and rsi.iloc[i] > rsi.iloc[i - 1]:
            buy_score += 1
        # Above SMA
        if df["Close"].iloc[i] > sma_20.iloc[i] > sma_50.iloc[i]:
            buy_score += 1
        # Volume confirmation
        if df["Volume"].iloc[i] > vol_ma.iloc[i] * 1.2:
            buy_score += 1

        if buy_score >= 3:
            signals.iloc[i] = 1

        # Sell conditions
        sell_score = 0
        if macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i - 1] >= signal_line.iloc[i - 1]:
            sell_score += 2
        if rsi.iloc[i] > 75:
            sell_score += 1
        if df["Close"].iloc[i] < sma_20.iloc[i]:
            sell_score += 1

        if sell_score >= 2:
            signals.iloc[i] = -1

    return signals


# ============================================================================
# AVAILABLE BACKTEST STRATEGIES (name -> func mapping)
# ============================================================================

BACKTEST_STRATEGIES = {
    "MACD Crossover": macd_crossover_signals,
    "RSI Mean Reversion": rsi_strategy_signals,
    "SMA Crossover (20/50)": sma_crossover_signals,
    "Bollinger Band Bounce": bollinger_band_signals,
    "ADX/DMI Trend": adx_dmi_signals,
    "Supertrend": supertrend_signals,
    "Combined Multi-Indicator": combined_strategy_signals,
}
