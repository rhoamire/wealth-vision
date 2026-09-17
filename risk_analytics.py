"""
Advanced Risk Analytics Module
Sharpe, Sortino, Max Drawdown, Beta, VaR, Correlation, and more
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def _strip_tz(series_or_df):
    """Remove timezone info from DatetimeIndex to avoid tz-naive vs tz-aware join errors."""
    if series_or_df is None:
        return series_or_df
    if hasattr(series_or_df, 'index') and hasattr(series_or_df.index, 'tz') and series_or_df.index.tz is not None:
        series_or_df = series_or_df.copy()
        series_or_df.index = series_or_df.index.tz_localize(None)
    return series_or_df


# ============================================================================
# RISK METRICS CALCULATOR
# ============================================================================

class RiskAnalytics:
    """Calculate comprehensive risk metrics for stocks and portfolios"""

    def __init__(self, risk_free_rate=0.06):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 6% for India)
        """
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _ensure_series(df_or_series, col="Close"):
        """Ensure data is a 1-D pandas Series (handles multi-level columns from yf.download)"""
        if df_or_series is None:
            return None
        if isinstance(df_or_series, pd.Series):
            return _strip_tz(df_or_series)
        if isinstance(df_or_series, pd.DataFrame):
            # Flatten multi-level columns if present
            if isinstance(df_or_series.columns, pd.MultiIndex):
                df_or_series = df_or_series.copy()
                df_or_series.columns = df_or_series.columns.get_level_values(0)
            if col in df_or_series.columns:
                s = df_or_series[col]
                if isinstance(s, pd.DataFrame):
                    return _strip_tz(s.iloc[:, 0])
                return _strip_tz(s)
            # Fallback: return the first column
            return _strip_tz(df_or_series.iloc[:, 0])
        return _strip_tz(pd.Series(df_or_series))

    def compute_all_metrics(self, df, benchmark_df=None):
        """Compute all risk metrics for a stock"""
        if df is None or len(df) < 30:
            return None

        close = self._ensure_series(df, "Close")
        if close is None or len(close) < 30:
            return None

        returns = close.pct_change().dropna()

        metrics = {}
        metrics.update(self._return_metrics(returns))
        metrics.update(self._volatility_metrics(returns))
        metrics.update(self._risk_adjusted_metrics(returns))
        metrics.update(self._drawdown_metrics(close))
        metrics.update(self._var_metrics(returns))
        metrics.update(self._tail_risk_metrics(returns))

        if benchmark_df is not None and len(benchmark_df) > 30:
            bench_close = self._ensure_series(benchmark_df, "Close")
            if bench_close is not None and len(bench_close) > 30:
                bench_returns = bench_close.pct_change().dropna()
                metrics.update(self._relative_metrics(returns, bench_returns))

        return metrics

    # ---- Return Metrics ----

    def _return_metrics(self, returns):
        """Calculate return-based metrics"""
        ann_return = returns.mean() * 252
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

        # Monthly returns stats
        monthly_returns = returns.resample("ME").sum() if hasattr(returns.index, "freq") else returns.groupby(
            pd.Grouper(freq="ME")).sum()
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        total_months = len(monthly_returns)

        return {
            "annualized_return": ann_return * 100,
            "total_return": total_return * 100,
            "cagr": cagr * 100,
            "best_day": returns.max() * 100,
            "worst_day": returns.min() * 100,
            "avg_daily_return": returns.mean() * 100,
            "positive_days_pct": (returns > 0).mean() * 100,
            "positive_months": int(positive_months),
            "negative_months": int(negative_months),
            "win_rate": positive_months / total_months * 100 if total_months > 0 else 0,
        }

    # ---- Volatility Metrics ----

    def _volatility_metrics(self, returns):
        """Calculate volatility-based metrics"""
        daily_vol = returns.std()
        ann_vol = daily_vol * np.sqrt(252)

        # Rolling volatility
        vol_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        vol_60d = returns.rolling(60).std().iloc[-1] * np.sqrt(252) if len(returns) > 60 else ann_vol

        # Downside volatility (only negative returns)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Upside volatility
        upside_returns = returns[returns > 0]
        upside_vol = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0

        return {
            "daily_volatility": daily_vol * 100,
            "annualized_volatility": ann_vol * 100,
            "volatility_20d": vol_20d * 100,
            "volatility_60d": vol_60d * 100,
            "downside_volatility": downside_vol * 100,
            "upside_volatility": upside_vol * 100,
            "upside_downside_ratio": upside_vol / downside_vol if downside_vol > 0 else float("inf"),
        }

    # ---- Risk-Adjusted Metrics ----

    def _risk_adjusted_metrics(self, returns):
        """Calculate risk-adjusted performance metrics"""
        excess = returns.mean() * 252 - self.risk_free_rate
        vol = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        sharpe = excess / vol if vol > 0 else 0

        # Sortino Ratio (downside deviation only)
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
        sortino = excess / downside_std if downside_std > 0 else 0

        # Calmar Ratio (return / max drawdown)
        ann_return = returns.mean() * 252
        prices = (1 + returns).cumprod()
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        max_dd = abs(drawdown.min())
        calmar = ann_return / max_dd if max_dd > 0 else 0

        # Information Ratio placeholder (needs benchmark)
        # Treynor Ratio placeholder (needs beta)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
        }

    # ---- Drawdown Metrics ----

    def _drawdown_metrics(self, prices):
        """Calculate drawdown-related metrics"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        max_dd = drawdown.min()

        # Current drawdown
        current_dd = drawdown.iloc[-1]

        # Drawdown duration
        in_drawdown = drawdown < 0
        dd_groups = (~in_drawdown).cumsum()
        if in_drawdown.any():
            dd_durations = in_drawdown.groupby(dd_groups).sum()
            max_dd_duration = dd_durations.max()
            avg_dd_duration = dd_durations[dd_durations > 0].mean()
        else:
            max_dd_duration = 0
            avg_dd_duration = 0

        # Recovery time from max drawdown
        max_dd_idx = drawdown.idxmin()
        recovery_data = drawdown.loc[max_dd_idx:]
        recovered = recovery_data[recovery_data >= 0]
        recovery_days = (recovered.index[0] - max_dd_idx).days if len(recovered) > 0 else None

        return {
            "max_drawdown": max_dd * 100,
            "current_drawdown": current_dd * 100,
            "max_drawdown_duration": int(max_dd_duration),
            "avg_drawdown_duration": float(avg_dd_duration),
            "recovery_days": recovery_days,
            "drawdown_series": drawdown,
        }

    # ---- Value at Risk ----

    def _var_metrics(self, returns, confidence_levels=None):
        """Calculate Value at Risk metrics"""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        var_results = {}
        for cl in confidence_levels:
            # Historical VaR
            var_hist = np.percentile(returns, (1 - cl) * 100) * 100
            var_results[f"var_{int(cl*100)}_historical"] = var_hist

            # Parametric VaR (assuming normal distribution)
            z_score = stats.norm.ppf(1 - cl)
            var_param = (returns.mean() + z_score * returns.std()) * 100
            var_results[f"var_{int(cl*100)}_parametric"] = var_param

            # Conditional VaR (Expected Shortfall)
            threshold = np.percentile(returns, (1 - cl) * 100)
            cvar = returns[returns <= threshold].mean() * 100
            var_results[f"cvar_{int(cl*100)}"] = cvar

        return var_results

    # ---- Tail Risk ----

    def _tail_risk_metrics(self, returns):
        """Calculate tail risk metrics"""
        # Skewness
        skewness = returns.skew()

        # Kurtosis (excess)
        kurtosis = returns.kurtosis()

        # Tail ratio
        pct_95 = np.percentile(returns, 95)
        pct_5 = abs(np.percentile(returns, 5))
        tail_ratio = abs(pct_95 / pct_5) if pct_5 > 0 else 0

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "tail_ratio": tail_ratio,
        }

    # ---- Relative / Benchmark Metrics ----

    def _relative_metrics(self, returns, benchmark_returns):
        """Calculate relative-to-benchmark metrics"""
        # Strip timezone info to avoid tz-naive vs tz-aware join errors
        returns = _strip_tz(returns)
        benchmark_returns = _strip_tz(benchmark_returns)
        # Align dates
        combined = pd.DataFrame({
            "stock": returns, "benchmark": benchmark_returns
        }).dropna()

        if len(combined) < 30:
            return {}

        stock_ret = combined["stock"]
        bench_ret = combined["benchmark"]

        # Beta
        covariance = np.cov(stock_ret, bench_ret)[0][1]
        bench_var = bench_ret.var()
        beta = covariance / bench_var if bench_var > 0 else 1

        # Alpha (Jensen's Alpha)
        stock_ann = stock_ret.mean() * 252
        bench_ann = bench_ret.mean() * 252
        alpha = stock_ann - (self.risk_free_rate + beta * (bench_ann - self.risk_free_rate))

        # Treynor Ratio
        excess = stock_ann - self.risk_free_rate
        treynor = excess / beta if beta != 0 else 0

        # Information Ratio
        active_returns = stock_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

        # Up/Down Capture
        up_market = bench_ret > 0
        down_market = bench_ret < 0
        up_capture = (stock_ret[up_market].mean() / bench_ret[up_market].mean() * 100
                      if bench_ret[up_market].mean() != 0 else 0)
        down_capture = (stock_ret[down_market].mean() / bench_ret[down_market].mean() * 100
                        if bench_ret[down_market].mean() != 0 else 0)

        # R-squared
        correlation = stock_ret.corr(bench_ret)
        r_squared = correlation ** 2

        return {
            "beta": beta,
            "alpha": alpha * 100,
            "treynor_ratio": treynor,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error * 100,
            "up_capture": up_capture,
            "down_capture": down_capture,
            "r_squared": r_squared * 100,
            "correlation_with_benchmark": correlation,
        }


# ============================================================================
# PORTFOLIO RISK ANALYZER
# ============================================================================

class PortfolioRiskAnalyzer:
    """Analyze risk for a portfolio of stocks"""

    def __init__(self, risk_free_rate=0.06):
        self.risk_analytics = RiskAnalytics(risk_free_rate)

    def analyze_portfolio_risk(self, stock_data_dict, weights=None):
        """
        Analyze portfolio risk.
        stock_data_dict: {symbol: DataFrame}
        weights: {symbol: weight} (should sum to 1)
        """
        if not stock_data_dict:
            return None

        symbols = list(stock_data_dict.keys())

        if weights is None:
            equal_weight = 1.0 / len(symbols)
            weights = {s: equal_weight for s in symbols}

        # Build returns matrix
        returns_dict = {}
        for symbol, df in stock_data_dict.items():
            if df is not None and len(df) > 30:
                close = RiskAnalytics._ensure_series(df, "Close")
                if close is not None:
                    s = _strip_tz(close.pct_change().dropna())
                    returns_dict[symbol] = s

        if len(returns_dict) < 2:
            return None

        returns_df = pd.DataFrame(returns_dict).dropna()
        w = np.array([weights.get(s, 0) for s in returns_df.columns])

        # Portfolio returns
        portfolio_returns = (returns_df * w).sum(axis=1)

        # Correlation matrix
        corr_matrix = returns_df.corr()

        # Covariance matrix (annualized)
        cov_matrix = returns_df.cov() * 252

        # Portfolio variance and volatility
        portfolio_var = np.dot(w, np.dot(cov_matrix, w))
        portfolio_vol = np.sqrt(portfolio_var)

        # Portfolio Sharpe
        portfolio_ann_return = portfolio_returns.mean() * 252
        portfolio_sharpe = ((portfolio_ann_return - self.risk_analytics.risk_free_rate) /
                            portfolio_vol if portfolio_vol > 0 else 0)

        # Diversification ratio
        weighted_vol = sum(w[i] * returns_df.iloc[:, i].std() * np.sqrt(252)
                          for i in range(len(w)))
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1

        # Marginal contribution to risk
        marginal_contrib = np.dot(cov_matrix, w) / portfolio_vol if portfolio_vol > 0 else np.zeros(len(w))

        # Component VaR
        z_95 = stats.norm.ppf(0.05)
        portfolio_var_95 = -(portfolio_returns.mean() + z_95 * portfolio_returns.std())
        component_var = {
            s: w[i] * marginal_contrib[i] * z_95
            for i, s in enumerate(returns_df.columns)
        }

        return {
            "portfolio_return": portfolio_ann_return * 100,
            "portfolio_volatility": portfolio_vol * 100,
            "portfolio_sharpe": portfolio_sharpe,
            "diversification_ratio": diversification_ratio,
            "correlation_matrix": corr_matrix,
            "covariance_matrix": cov_matrix,
            "marginal_risk_contribution": dict(zip(returns_df.columns, marginal_contrib)),
            "portfolio_var_95": portfolio_var_95 * 100,
            "component_var": component_var,
            "portfolio_returns_series": portfolio_returns,
            "individual_metrics": {
                s: self.risk_analytics.compute_all_metrics(stock_data_dict[s])
                for s in symbols if s in stock_data_dict and stock_data_dict[s] is not None
            },
        }

    def efficient_frontier(self, stock_data_dict, n_portfolios=5000):
        """Generate efficient frontier points"""
        symbols = list(stock_data_dict.keys())
        returns_dict = {}
        for s, df in stock_data_dict.items():
            if df is not None and len(df) > 30:
                close = RiskAnalytics._ensure_series(df, "Close")
                if close is not None:
                    returns_dict[s] = _strip_tz(close.pct_change().dropna())

        returns_df = pd.DataFrame(returns_dict).dropna()
        n = len(returns_df.columns)
        if n < 2:
            return None

        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252

        results = []
        for _ in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n))
            ret = np.dot(w, mean_returns)
            vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            sharpe = ((ret - self.risk_analytics.risk_free_rate) / vol) if vol > 0 else 0
            results.append({
                "return": ret * 100,
                "volatility": vol * 100,
                "sharpe": sharpe,
                "weights": dict(zip(returns_df.columns, w)),
            })

        results_df = pd.DataFrame(results)

        # Find optimal portfolios
        max_sharpe_idx = results_df["sharpe"].idxmax()
        min_vol_idx = results_df["volatility"].idxmin()

        return {
            "portfolios": results_df,
            "max_sharpe": results_df.iloc[max_sharpe_idx].to_dict(),
            "min_volatility": results_df.iloc[min_vol_idx].to_dict(),
        }


# ============================================================================
# STOCK COMPARISON
# ============================================================================

class StockComparator:
    """Compare multiple stocks on various metrics"""

    def __init__(self, risk_free_rate=0.06):
        self.risk_analytics = RiskAnalytics(risk_free_rate)

    def compare(self, stock_data_dict, benchmark_df=None):
        """Compare multiple stocks"""
        comparison = {}
        for symbol, df in stock_data_dict.items():
            if df is None or len(df) < 30:
                continue
            try:
                metrics = self.risk_analytics.compute_all_metrics(df, benchmark_df)
                if metrics:
                    # Remove series data for comparison table
                    clean = {k: v for k, v in metrics.items()
                             if not isinstance(v, (pd.Series, pd.DataFrame))}
                    comparison[symbol] = clean
            except Exception:
                pass

        return pd.DataFrame(comparison).T if comparison else pd.DataFrame()

    def rank_stocks(self, comparison_df, criteria=None):
        """Rank stocks by multiple criteria"""
        if comparison_df.empty:
            return comparison_df

        if criteria is None:
            criteria = {
                "sharpe_ratio": {"weight": 0.3, "ascending": False},
                "sortino_ratio": {"weight": 0.2, "ascending": False},
                "max_drawdown": {"weight": 0.2, "ascending": True},  # Less negative is better
                "annualized_return": {"weight": 0.15, "ascending": False},
                "annualized_volatility": {"weight": 0.15, "ascending": True},
            }

        rank_scores = pd.Series(0.0, index=comparison_df.index)
        for metric, params in criteria.items():
            if metric in comparison_df.columns:
                ranked = comparison_df[metric].rank(ascending=params["ascending"])
                rank_scores += ranked * params["weight"]

        comparison_df["composite_rank_score"] = rank_scores
        return comparison_df.sort_values("composite_rank_score", ascending=False)
