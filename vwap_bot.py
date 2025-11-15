"""Simple VWAP crossover bot skeleton
Place this file in the trading_setup folder. It is a runnable skeleton in dry-run mode.
Fill .env with credentials and settings before running live.
"""
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load env from same folder
HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    # fall back to environment
    pass

# config from env
SYMBOL = os.getenv("SYMBOL", "BTCUSD")
TIMEFRAME_MINUTES = int(os.getenv("TIMEFRAME_MINUTES", "60"))
SMA_LENGTH = int(os.getenv("SMA_LENGTH", "20"))
POSITION_LOTS = int(os.getenv("POSITION_LOTS", "2"))
ORDER_SIZE_USD = float(os.getenv("ORDER_SIZE_USD", "10.0"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
WEBAPI_URL = os.getenv("WEBAPI_URL", "https://api.delta.exchange")
CANDLE_CLOSE_DELAY_SEC = int(os.getenv("CANDLE_CLOSE_DELAY_SEC", "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------- indicators ---------------------------
def compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp")
    d["date"] = d["timestamp"].dt.tz_convert("UTC").dt.date
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    d["tpv"] = tp * d["volume"]
    d["cum_tpv"] = d.groupby("date")["tpv"].cumsum()
    d["cum_vol"] = d.groupby("date")["volume"].cumsum()
    vwap = d["cum_tpv"] / d["cum_vol"].replace(0, np.nan)
    return vwap.fillna(method="ffill").fillna(d["close"])


def compute_sma_close(df: pd.DataFrame, length: int) -> pd.Series:
    return df["close"].rolling(window=length, min_periods=1).mean()


def compute_calculated_line(df: pd.DataFrame, vwap: pd.Series, sma_close: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        calc = (vwap * sma_close) / df["close"]
    return calc


@dataclass
class Signal:
    side: Optional[str]
    entry_price: Optional[float]
    sl_price: Optional[float]
    info: dict


def detect_crossover_signal(df: pd.DataFrame) -> Signal:
    if len(df) < 3:
        return Signal(None, None, None, {"reason": "not enough candles"})
    last = len(df) - 1
    prev = last - 1
    vwap = df["_vwap"].values
    calc = df["_calc"].values
    # bullish crossover
    if (vwap[prev] < calc[prev]) and (vwap[last] > calc[last]):
        sl = df.loc[last, "low"]
        entry = df.loc[last, "close"]
        return Signal("long", entry, sl, {"crossover_at": str(df.loc[last, "timestamp"])})
    if (vwap[prev] > calc[prev]) and (vwap[last] < calc[last]):
        sl = df.loc[last, "high"]
        entry = df.loc[last, "close"]
        return Signal("short", entry, sl, {"crossover_at": str(df.loc[last, "timestamp"])})
    return Signal(None, None, None, {"reason": "no crossover"})


# --------------------------- Broker stub ---------------------------
class Broker:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], dry_run=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run

    def close_all_positions(self, symbol: str):
        logger.info("close_all_positions(%s) dry_run=%s", symbol, self.dry_run)
        if self.dry_run:
            return True
        raise NotImplementedError("Implement with Delta API")

    def place_market_order(self, symbol: str, side: str, size, stop_loss=None):
        logger.info("place_market_order %s %s size=%s stop_loss=%s dry_run=%s", symbol, side, size, stop_loss, self.dry_run)
        if self.dry_run:
            return {"order_id": f"dryrun-{int(time.time())}", "status": "filled", "filled_price": size}
        raise NotImplementedError("Implement with Delta API")


# --------------------------- Runner ---------------------------
def run_once_with_candles(df: pd.DataFrame, broker: Broker):
    df = df.copy()
    # normalize
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise ValueError("candles must include timestamp column")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_vwap"] = compute_daily_vwap(df)
    df["_sma_close"] = compute_sma_close(df, SMA_LENGTH)
    df["_calc"] = compute_calculated_line(df, df["_vwap"], df["_sma_close"])
    signal = detect_crossover_signal(df)
    logger.info("Signal: %s %s", signal.side, signal.info)
    if signal.side is None:
        return None
    # close all positions then enter
    broker.close_all_positions(SYMBOL)
    side = "buy" if signal.side == "long" else "sell"
    # size: using POSITION_LOTS as surrogate size for now
    order = broker.place_market_order(SYMBOL, side, size=POSITION_LOTS, stop_loss=signal.sl_price)
    return order


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", help="CSV file with hourly candles (timestamp,open,high,low,close,volume)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    broker = Broker(os.getenv("DELTA_API_KEY"), os.getenv("DELTA_API_SECRET"), dry_run=DRY_RUN)

    if args.candles:
        df = pd.read_csv(args.candles)
        res = run_once_with_candles(df, broker)
        print(res)
    else:
        logger.info("No --candles provided. Implement a live fetcher to use this script in continuous mode.")
