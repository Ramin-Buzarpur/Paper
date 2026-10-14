from __future__ import annotations

MARKET_CATEGORIES: dict[str, dict[str, str]] = {
    "Forex Majors": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X",
    },
    "Cryptocurrency": {
        "Bitcoin (BTC/USD)": "BTC-USD",
        "Ethereum (ETH/USD)": "ETH-USD",
        "Solana (SOL/USD)": "SOL-USD",
        "Binance Coin (BNB/USD)": "BNB-USD",
        "Ripple (XRP/USD)": "XRP-USD",
        "Cardano (ADA/USD)": "ADA-USD",
    },
    "Commodities": {
        "Brent Crude Oil": "BZ=F",
        "WTI Crude Oil": "CL=F",
        "Gold (Comex)": "GC=F",
        "Silver (Comex)": "SI=F",
        "Copper": "HG=F",
        "Natural Gas": "NG=F",
    },
    "Global Indices": {
        "S&P 500 (US)": "^GSPC",
        "Nasdaq 100 (US Tech)": "^NDX",
        "Dow Jones 30 (US Industrial)": "^DJI",
        "DAX 40 (Germany)": "^GDAXI",
        "FTSE 100 (UK)": "^FTSE",
        "Nikkei 225 (Japan)": "^N225",
    },
}

# crypto trades every day, everything else only on business days
CATEGORY_CALENDAR: dict[str, str] = {
    "Forex Majors": "B",
    "Cryptocurrency": "D",
    "Commodities": "B",
    "Global Indices": "B",
}
