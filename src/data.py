import numpy as np
import yfinance as yf

from src.config import START_DATE, END_DATE, TRAIN_END, TEST_START, WF_WINDOWS

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "JNJ",
    "WMT", "XOM", "UNH", "MA", "PG", "HD", "CVX", "MRK", "ABBV", "KO",
    "PEP", "COST", "AVGO", "MCD", "CSCO", "ACN", "TMO", "ADBE", "ABT", "NKE",
    "LLY", "DHR", "TXN", "NEE", "VZ", "CMCSA", "ORCL", "PM", "WFC", "BMY",
    "UPS", "RTX", "HON", "QCOM", "LOW", "AMGN", "INTC", "UNP", "BA", "SPGI",
    "CAT", "GE", "IBM", "SBUX", "COP", "AXP", "MMM", "GS", "BLK", "PLD",
    "AMD", "ISRG", "GILD", "AMT", "MDLZ", "NOW", "ADI", "DE", "SYK", "PFE",
    "TJX", "BKNG", "MO", "CI", "ADP", "VRTX", "ZTS", "LRCX", "CME", "MMC",
    "CB", "SHW", "SO", "DUK", "PGR", "TGT", "ITW", "BSX", "ETN", "CL",
    "APD", "EOG", "NSC", "HUM", "NOC", "BDX", "REGN", "ICE", "EQIX", "SLB"
]


def load_data():
    data = yf.download(TICKERS, START_DATE, END_DATE, group_by="ticker")
    close = data.xs("Close", axis=1, level=1)
    close = close.dropna(how='all')
    logreturn = (np.log(close).diff()).dropna(how='all')
    return logreturn


def train_test_split(logreturn):
    train = np.asarray(logreturn.loc[:TRAIN_END].dropna(how='all'))
    test = np.asarray(logreturn.loc[TEST_START:].dropna(how='all'))
    return train, test


def walk_forward_splits(logreturn):
    """Return (train, dev) numpy array pairs for each expanding walk-forward window.

    Training always starts from START_DATE; the window expands fold by fold.
    Dev periods are non-overlapping and come immediately after each training
    window, so no future data ever leaks into training.
    """
    splits = []
    for train_end, dev_start, dev_end in WF_WINDOWS:
        train = np.asarray(logreturn.loc[:train_end].dropna(how='all'))
        dev = np.asarray(logreturn.loc[dev_start:dev_end].dropna(how='all'))
        splits.append((train, dev))
    return splits
