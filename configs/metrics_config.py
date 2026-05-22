"""Configuration for metrics/calculator.py. Edit these values per strategy."""

STARTING_CAPITAL = 1_000_000                                          # INR (10 lakhs)
RISK_FREE_RATE_ANNUAL = 0.06                                          # Indian 10-year govt bond yield
TRADING_DAYS_PER_YEAR = 252                                           # Indian market trading days per year
DAILY_RISK_FREE_RATE = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / TRADING_DAYS_PER_YEAR) - 1
PROFIT_FACTOR_SUSPICIOUS_THRESHOLD = 4.0
