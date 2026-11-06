import yfinance as yf
import pandas as pd

# Download NVIDIA historical stock data
nvda = yf.download(
    "NVDA",
    start="2020-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# Display the first 5 rows
nvda.head()
