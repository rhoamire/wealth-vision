"""
Fundamental Analysis Module
Fetch and compute fundamental metrics from yfinance
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# FUNDAMENTAL ANALYSIS
# ============================================================================

class FundamentalAnalyzer:
    """Retrieve and compute fundamental analysis metrics for stocks"""

    # Scoring thresholds for different ratios
    VALUATION_THRESHOLDS = {
        "pe_ratio": {"cheap": 15, "fair": 25, "expensive": 40},
        "pb_ratio": {"cheap": 1.5, "fair": 3, "expensive": 5},
        "peg_ratio": {"cheap": 1, "fair": 1.5, "expensive": 2},
        "ev_ebitda": {"cheap": 10, "fair": 15, "expensive": 25},
    }

    def get_fundamentals(self, symbol, exchange="NSE"):
        """Get comprehensive fundamental data for a stock"""
        try:
            suffix = {"NSE": ".NS", "BSE": ".BO", "NYSE": "", "NASDAQ": "", "LSE": ".L"}
            full_symbol = f"{symbol}{suffix.get(exchange, '')}"
            ticker = yf.Ticker(full_symbol)
            info = ticker.info

            if not info or info.get("regularMarketPrice") is None:
                return None

            # Valuation metrics
            valuation = {
                "pe_ratio": info.get("trailingPE", info.get("forwardPE", None)),
                "forward_pe": info.get("forwardPE", None),
                "pb_ratio": info.get("priceToBook", None),
                "ps_ratio": info.get("priceToSalesTrailing12Months", None),
                "peg_ratio": info.get("pegRatio", None),
                "ev_revenue": info.get("enterpriseToRevenue", None),
                "ev_ebitda": info.get("enterpriseToEbitda", None),
                "market_cap": info.get("marketCap", None),
                "enterprise_value": info.get("enterpriseValue", None),
            }

            # Profitability metrics
            profitability = {
                "profit_margin": info.get("profitMargins", None),
                "operating_margin": info.get("operatingMargins", None),
                "gross_margin": info.get("grossMargins", None),
                "roe": info.get("returnOnEquity", None),
                "roa": info.get("returnOnAssets", None),
                "ebitda_margin": None,
            }
            # Compute EBITDA margin if possible
            ebitda = info.get("ebitda", None)
            revenue = info.get("totalRevenue", None)
            if ebitda and revenue and revenue > 0:
                profitability["ebitda_margin"] = ebitda / revenue

            # Growth metrics
            growth = {
                "revenue_growth": info.get("revenueGrowth", None),
                "earnings_growth": info.get("earningsGrowth", None),
                "quarterly_revenue_growth": info.get("revenueQuarterlyGrowth", None),
                "quarterly_earnings_growth": info.get("earningsQuarterlyGrowth", None),
            }

            # Financial health
            health = {
                "current_ratio": info.get("currentRatio", None),
                "quick_ratio": info.get("quickRatio", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "total_debt": info.get("totalDebt", None),
                "total_cash": info.get("totalCash", None),
                "free_cash_flow": info.get("freeCashflow", None),
                "operating_cash_flow": info.get("operatingCashflow", None),
            }

            # Dividend info
            dividends = {
                "dividend_yield": info.get("dividendYield", None),
                "dividend_rate": info.get("dividendRate", None),
                "payout_ratio": info.get("payoutRatio", None),
                "ex_dividend_date": info.get("exDividendDate", None),
                "five_year_avg_yield": info.get("fiveYearAvgDividendYield", None),
            }

            # Per-share metrics
            per_share = {
                "eps_trailing": info.get("trailingEps", None),
                "eps_forward": info.get("forwardEps", None),
                "book_value": info.get("bookValue", None),
                "revenue_per_share": info.get("revenuePerShare", None),
            }

            # Company info
            company_info = {
                "name": info.get("longName", info.get("shortName", symbol)),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "website": info.get("website", ""),
                "description": info.get("longBusinessSummary", ""),
                "employees": info.get("fullTimeEmployees", None),
                "country": info.get("country", "N/A"),
            }

            return {
                "symbol": symbol,
                "exchange": exchange,
                "company_info": company_info,
                "valuation": valuation,
                "profitability": profitability,
                "growth": growth,
                "financial_health": health,
                "dividends": dividends,
                "per_share": per_share,
                "current_price": info.get("regularMarketPrice", info.get("currentPrice", None)),
                "52w_high": info.get("fiftyTwoWeekHigh", None),
                "52w_low": info.get("fiftyTwoWeekLow", None),
            }

        except Exception as e:
            print(f"Fundamental analysis error for {symbol}: {e}")
            return None

    def score_fundamentals(self, fundamentals):
        """Score a stock's fundamentals on a 0-100 scale"""
        if not fundamentals:
            return None

        scores = {}
        total_score = 0
        total_weight = 0

        val = fundamentals.get("valuation", {})
        prof = fundamentals.get("profitability", {})
        growth = fundamentals.get("growth", {})
        health = fundamentals.get("financial_health", {})
        div = fundamentals.get("dividends", {})

        # Valuation score (weight: 25)
        val_score = 50
        pe = val.get("pe_ratio")
        if pe is not None and pe > 0:
            if pe < 15:
                val_score = 85
            elif pe < 25:
                val_score = 65
            elif pe < 40:
                val_score = 45
            else:
                val_score = 25

        pb = val.get("pb_ratio")
        if pb is not None and pb > 0:
            pb_score = 80 if pb < 1.5 else (60 if pb < 3 else (40 if pb < 5 else 20))
            val_score = (val_score + pb_score) / 2

        scores["valuation"] = val_score
        total_score += val_score * 25
        total_weight += 25

        # Profitability score (weight: 25)
        prof_score = 50
        roe = prof.get("roe")
        if roe is not None:
            roe_pct = roe * 100 if roe < 1 else roe
            prof_score = min(max(roe_pct * 3, 10), 95)

        margin = prof.get("profit_margin")
        if margin is not None:
            margin_pct = margin * 100 if margin < 1 else margin
            margin_score = min(max(margin_pct * 3, 10), 90)
            prof_score = (prof_score + margin_score) / 2

        scores["profitability"] = prof_score
        total_score += prof_score * 25
        total_weight += 25

        # Growth score (weight: 20)
        growth_score = 50
        rev_growth = growth.get("revenue_growth")
        earn_growth = growth.get("earnings_growth")

        if rev_growth is not None:
            rg = rev_growth * 100 if abs(rev_growth) < 5 else rev_growth
            growth_score = min(max(rg * 2 + 50, 10), 95)

        if earn_growth is not None:
            eg = earn_growth * 100 if abs(earn_growth) < 5 else earn_growth
            eg_score = min(max(eg * 2 + 50, 10), 95)
            growth_score = (growth_score + eg_score) / 2

        scores["growth"] = growth_score
        total_score += growth_score * 20
        total_weight += 20

        # Financial health score (weight: 20)
        health_score = 50
        de = health.get("debt_to_equity")
        if de is not None:
            health_score = 85 if de < 50 else (65 if de < 100 else (45 if de < 200 else 25))

        cr = health.get("current_ratio")
        if cr is not None:
            cr_score = 80 if cr > 2 else (60 if cr > 1.5 else (40 if cr > 1 else 20))
            health_score = (health_score + cr_score) / 2

        scores["financial_health"] = health_score
        total_score += health_score * 20
        total_weight += 20

        # Dividend score (weight: 10)
        div_score = 50
        dy = div.get("dividend_yield")
        if dy is not None:
            dy_pct = dy * 100 if dy < 1 else dy
            div_score = min(max(dy_pct * 15 + 30, 10), 90)

        scores["dividend"] = div_score
        total_score += div_score * 10
        total_weight += 10

        overall = total_score / total_weight if total_weight > 0 else 50

        # Rating
        if overall >= 75:
            rating = "Strong Buy"
        elif overall >= 60:
            rating = "Buy"
        elif overall >= 45:
            rating = "Hold"
        elif overall >= 30:
            rating = "Sell"
        else:
            rating = "Strong Sell"

        return {
            "overall_score": overall,
            "rating": rating,
            "category_scores": scores,
        }

    def compare_fundamentals(self, symbol_list, exchange="NSE"):
        """Compare fundamentals of multiple stocks"""
        comparison = {}
        for symbol in symbol_list:
            fund = self.get_fundamentals(symbol, exchange)
            if fund:
                score_result = self.score_fundamentals(fund)
                flat = {
                    "P/E": fund["valuation"].get("pe_ratio"),
                    "P/B": fund["valuation"].get("pb_ratio"),
                    "EV/EBITDA": fund["valuation"].get("ev_ebitda"),
                    "ROE %": (fund["profitability"].get("roe") or 0) * 100,
                    "Profit Margin %": (fund["profitability"].get("profit_margin") or 0) * 100,
                    "Revenue Growth %": (fund["growth"].get("revenue_growth") or 0) * 100,
                    "D/E": fund["financial_health"].get("debt_to_equity"),
                    "Current Ratio": fund["financial_health"].get("current_ratio"),
                    "Div Yield %": (fund["dividends"].get("dividend_yield") or 0) * 100,
                    "Score": score_result["overall_score"] if score_result else None,
                    "Rating": score_result["rating"] if score_result else "N/A",
                }
                comparison[symbol] = flat

        return pd.DataFrame(comparison).T if comparison else pd.DataFrame()

    def get_financial_statements(self, symbol, exchange="NSE"):
        """Get income statement, balance sheet, and cash flow data"""
        try:
            suffix = {"NSE": ".NS", "BSE": ".BO", "NYSE": "", "NASDAQ": "", "LSE": ".L"}
            full_symbol = f"{symbol}{suffix.get(exchange, '')}"
            ticker = yf.Ticker(full_symbol)

            return {
                "income_statement": ticker.financials if hasattr(ticker, "financials") else None,
                "quarterly_income": ticker.quarterly_financials if hasattr(ticker, "quarterly_financials") else None,
                "balance_sheet": ticker.balance_sheet if hasattr(ticker, "balance_sheet") else None,
                "cash_flow": ticker.cashflow if hasattr(ticker, "cashflow") else None,
            }
        except Exception:
            return None
