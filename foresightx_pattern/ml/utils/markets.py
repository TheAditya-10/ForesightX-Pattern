from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketProfile:
    timezone: str
    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int


US_EQUITY_MARKET = MarketProfile(
    timezone="America/New_York",
    open_hour=9,
    open_minute=30,
    close_hour=16,
    close_minute=0,
)

INDIA_EQUITY_MARKET = MarketProfile(
    timezone="Asia/Kolkata",
    open_hour=9,
    open_minute=15,
    close_hour=15,
    close_minute=30,
)


def market_profile_for_ticker(ticker: str) -> MarketProfile:
    symbol = ticker.upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return INDIA_EQUITY_MARKET
    return US_EQUITY_MARKET
