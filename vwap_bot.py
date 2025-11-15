"""Simple VWAP crossover bot skeleton
Place this file in the trading_setup folder. It is a runnable skeleton in dry-run mode.
Fill .env with credentials and settings before running live.
"""
import os
import time
import logging
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import requests
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
WEBAPI_URL = os.getenv("WEBAPI_URL", "https://api.india.delta.exchange")
CANDLE_CLOSE_DELAY_SEC = int(os.getenv("CANDLE_CLOSE_DELAY_SEC", "5"))

# Validate required environment variables for live trading
if not DRY_RUN:
    required_vars = ["DELTA_API_KEY", "DELTA_API_SECRET"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables for live trading: {missing_vars}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------- indicators ---------------------------
def compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute Volume Weighted Average Price (VWAP) for each day"""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp")
    d["date"] = d["timestamp"].dt.tz_convert("UTC").dt.date
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    d["tpv"] = tp * d["volume"]
    d["cum_tpv"] = d.groupby("date")["tpv"].cumsum()
    d["cum_vol"] = d.groupby("date")["volume"].cumsum()
    vwap = d["cum_tpv"] / d["cum_vol"].replace(0, np.nan)
    # Fixed deprecated fillna method
    return vwap.ffill().fillna(d["close"])


def compute_sma_close(df: pd.DataFrame, length: int) -> pd.Series:
    """Compute Simple Moving Average of close prices"""
    return df["close"].rolling(window=length, min_periods=1).mean()


def compute_calculated_line(df: pd.DataFrame, vwap: pd.Series, sma_close: pd.Series) -> pd.Series:
    """Compute calculated line: (VWAP * SMA) / Close"""
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
    """Detect crossover signals between VWAP and calculated line"""
    if len(df) < 3:
        return Signal(None, None, None, {"reason": "not enough candles"})
    
    last = len(df) - 1
    prev = last - 1
    vwap = df["_vwap"].values
    calc = df["_calc"].values
    
    # Bullish crossover: VWAP crosses above calculated line
    if (vwap[prev] <= calc[prev]) and (vwap[last] > calc[last]):
        sl = df.loc[last, "low"]
        entry = df.loc[last, "close"]
        return Signal("long", entry, sl, {"crossover_at": str(df.loc[last, "timestamp"])})
    
    # Bearish crossover: VWAP crosses below calculated line
    if (vwap[prev] >= calc[prev]) and (vwap[last] < calc[last]):
        sl = df.loc[last, "high"]
        entry = df.loc[last, "close"]
        return Signal("short", entry, sl, {"crossover_at": str(df.loc[last, "timestamp"])})
    
    return Signal(None, None, None, {"reason": "no crossover"})


# --------------------------- Delta Exchange Broker ---------------------------
class Broker:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], dry_run=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self.base_url = WEBAPI_URL
        
    def _generate_signature(self, secret: str, message: str) -> str:
        """Generate HMAC signature for API authentication"""
        message = bytes(message, 'utf-8')
        secret = bytes(secret, 'utf-8')
        hash = hmac.new(secret, message, hashlib.sha256)
        return hash.hexdigest()
    
    def _make_request(self, method: str, path: str, query_params: Optional[Dict] = None, payload: str = '') -> Optional[Dict]:
        """Make authenticated request to Delta Exchange API"""
        if self.dry_run:
            # We are modifying dry run to NOT return here, so we can test get_positions
            if method != 'GET' and path != '/v2/positions':
                 return {"status": "dry_run", "message": "Request not executed in dry run mode"}
            
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials required for live trading")
            
        timestamp = str(int(time.time()))
        url = f'{self.base_url}{path}'
        
        # Build query string
        query_string = ''
        if query_params:
            query_string = '&'.join([f'{k}={v}' for k, v in query_params.items()])
            query_string = '?' + query_string
        
        # Create signature
        signature_data = method + timestamp + path + query_string + payload
        signature = self._generate_signature(self.api_secret, signature_data)
        
        # Headers
        headers = {
            'api-key': self.api_key,
            'timestamp': timestamp,
            'signature': signature,
            'User-Agent': 'vwap-crossover-bot',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.request(
                method, 
                url, 
                data=payload, 
                params=query_params, 
                headers=headers, 
                timeout=(3, 27)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_product_id(self, symbol: str) -> Optional[int]:
        """Get product ID for a given symbol"""
        try:
            response = requests.get(f'{self.base_url}/v2/products', timeout=(3, 27))
            if response.status_code == 200:
                products = response.json().get('result', [])
                for product in products:
                    if product.get('symbol') == symbol:
                        return product.get('id')
            logger.error(f"Product not found for symbol: {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching product ID: {e}")
            return None
    
    # --- NEW FUNCTION TO GET LIVE CANDLES ---
    def get_candles(self, symbol: str, timeframe: int, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch historical candles for a symbol"""
        logger.info(f"Fetching {limit} candles for {symbol} on {timeframe}m timeframe...")
        
        product_id = self.get_product_id(symbol)
        if not product_id:
            logger.error(f"Cannot fetch candles: Product ID not found for {symbol}")
            return None
            
        # Delta's API uses 'resolution' in minutes
        params = {
            'product_id': product_id,
            'resolution': timeframe,
            'limit': limit
        }
        
        # This is a PUBLIC endpoint, so we use 'requests' directly, not '_make_request'
        try:
            url = f'{self.base_url}/v2/history/candles'
            response = requests.get(url, params=params, timeout=(3, 10))
            
            if response.status_code == 200:
                data = response.json().get('result', [])
                if not data:
                    logger.error("No candle data returned from API.")
                    return None
                    
                # Convert to DataFrame
                df = pd.DataFrame(data)
                # Ensure correct columns and types
                df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.astype(float)
                return df
            else:
                logger.error(f"Error fetching candles: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching candles: {e}")
            return None
    # --- END OF NEW FUNCTION ---

    def get_positions(self, symbol: str) -> Optional[Dict]:
        """Get current positions for a symbol"""
        product_id = self.get_product_id(symbol)
        if not product_id:
            return None
            
        response = self._make_request('GET', '/v2/positions', {'product_id': product_id})
        return response
    
    def close_all_positions(self, symbol: str) -> bool:
        """Close all positions for a given symbol"""
        logger.info("close_all_positions(%s) dry_run=%s", symbol, self.dry_run)
        
        # We need to get positions even in dry run to know what to do
        try:
            positions_data = self.get_positions(symbol)
            if not positions_data or not positions_data.get('result'):
                logger.info("No positions to close for %s", symbol)
                return True
                
            positions = positions_data['result']
            
            # If positions is not a list (e.g., error message), handle it
            if not isinstance(positions, list):
                logger.warning(f"Could not parse positions: {positions}")
                return True # Assume no positions

            for position in positions:
                if position.get('size', 0) != 0:
                    # Close position by placing opposite order
                    side = 'sell' if float(position['size']) > 0 else 'buy'
                    size = abs(float(position['size']))
                    
                    logger.info(f"Closing position: {side} {size} lots of {symbol}")
                    
                    if self.dry_run:
                        logger.info("DRY RUN: Would close position.")
                    else:
                        self.place_market_order(symbol, side, int(size))
            return True
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return False
    
    def place_market_order(self, symbol: str, side: str, size: int, stop_loss: Optional[float] = None) -> Optional[Dict]:
        """Place a market order"""
        logger.info("place_market_order %s %s size=%s stop_loss=%s dry_run=%s", 
                     symbol, side, size, stop_loss, self.dry_run)
        
        if self.dry_run:
            return {
                "order_id": f"dryrun-{int(time.time())}", 
                "status": "filled", 
                "symbol": symbol,
                "side": side,
                "size": size,
                "stop_loss": stop_loss
            }
        
        product_id = self.get_product_id(symbol)
        if not product_id:
            logger.error(f"Cannot place order: Product ID not found for {symbol}")
            return None
        
        # Validate lot size (must be integer)
        if not isinstance(size, int) or size <= 0:
            logger.error(f"Invalid lot size: {size}. Must be a positive integer.")
            return None
        
        order_data = {
            "product_id": product_id,
            "order_type": "market_order",
            "side": side,
            "size": size
        }
        
        payload = str(order_data).replace("'", '"')
        
        try:
            response = self._make_request('POST', '/v2/orders', payload=payload)
            
            if response and (response.get('success') or response.get('order_id')):
                logger.info(f"Order placed successfully: {response}")
                
                # Place stop loss order if specified
                if stop_loss:
                    self._place_stop_loss_order(product_id, side, size, stop_loss)
                    
                return response
            else:
                logger.error(f"Order placement failed: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _place_stop_loss_order(self, product_id: int, original_side: str, size: int, stop_price: float):
        """Place a stop loss order"""
        try:
            # Stop loss side is opposite to original order
            sl_side = 'sell' if original_side == 'buy' else 'buy'
            
            sl_order_data = {
                "product_id": product_id,
                "order_type": "stop_loss_order",
                "side": sl_side,
                "size": size,
                "stop_price": str(stop_price)
            }
            
            payload = str(sl_order_data).replace("'", '"')
            response = self._make_request('POST', '/v2/orders', payload=payload)
            
            if response and (response.get('success') or response.get('order_id')):
                logger.info(f"Stop loss order placed: {response}")
            else:
                logger.error(f"Stop loss order failed: {response}")
                
        except Exception as e:
            logger.error(f"Error placing stop loss order: {e}")


# --------------------------- Runner ---------------------------
def validate_candle_data(df: pd.DataFrame) -> bool:
    """Validate that candle data has required columns and is properly formatted"""
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logger.error(f"Missing required columns: {missing}")
        return False
    
    if len(df) < SMA_LENGTH:
        logger.error(f"Not enough data points. Need at least {SMA_LENGTH} candles for SMA calculation")
        return False
    
    # Check for null values in critical columns
    if df[required_columns].isnull().any().any():
        logger.error("Null values found in candle data")
        return False
    
    return True


def run_once_with_candles(df: pd.DataFrame, broker: Broker) -> Optional[Dict]:
    """Run the trading strategy once with provided candle data"""
    try:
        df = df.copy()
        
        # Validate input data
        if not validate_candle_data(df):
            return None
        
        # Normalize timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            raise ValueError("candles must include timestamp column")
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Calculate indicators
        df["_vwap"] = compute_daily_vwap(df)
        df["_sma_close"] = compute_sma_close(df, SMA_LENGTH)
        df["_calc"] = compute_calculated_line(df, df["_vwap"], df["_sma_close"])
        
        # Detect signal
        signal = detect_crossover_signal(df)
        logger.info("Signal: %s %s", signal.side, signal.info)
        
        if signal.side is None:
            return None
        
        # Execute trading logic
        # First close all existing positions
        if not broker.close_all_positions(SYMBOL):
            logger.error("Failed to close existing positions")
            return None
        
        # Wait a moment for positions to close
        if not broker.dry_run:
            time.sleep(1)
        
        # Place new order
        side = "buy" if signal.side == "long" else "sell"
        order = broker.place_market_order(SYMBOL, side, size=POSITION_LOTS, stop_loss=signal.sl_price)
        
        return order
        
    except Exception as e:
        logger.error(f"Error in run_once_with_candles: {e}", exc_info=True)
        return None

# --- NEW 24/7 LIVE LOOP ---
if __name__ == "__main__":
    # Initialize broker
    broker = Broker(
        os.getenv("DELTA_API_KEY"), 
        os.getenv("DELTA_API_SECRET"), 
        dry_run=DRY_RUN
    )
    
    logger.info("--- Starting VWAP Crossover Bot (LIVE) ---")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Timeframe: {TIMEFRAME_MINUTES} minutes")
    logger.info(f"SMA Length: {SMA_LENGTH}")
    logger.info(f"Position Size: {POSITION_LOTS} lots")
    logger.info(f"Dry Run: {DRY_RUN}")
    logger.info(f"API URL: {WEBAPI_URL}")

    # --- THIS IS THE MAIN BOT LOOP ---
    while True:
        try:
            # 1. Calculate when the next candle closes
            now = time.time()
            # Calculate the timestamp of the *next* candle's start/end
            next_candle_timestamp = (now // (TIMEFRAME_MINUTES * 60) + 1) * (TIMEFRAME_MINUTES * 60)
            
            # Wait until a few seconds *after* the candle closes
            wait_time = (next_candle_timestamp - now) + CANDLE_CLOSE_DELAY_SEC
            
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.2f} seconds for next candle close...")
                time.sleep(wait_time)
            
            # 2. Fetch the latest candle data
            df_candles = broker.get_candles(SYMBOL, TIMEFRAME_MINUTES)
            
            if df_candles is None or df_candles.empty:
                logger.warning("Could not fetch candle data. Retrying in 60s...")
                time.sleep(60)
                continue
                
            # 3. Run the strategy
            logger.info(f"Running strategy check with {len(df_candles)} candles...")
            result = run_once_with_candles(df_candles, broker)
            
            if result:
                logger.info(f"--- Trade Executed ---: {result}")
            else:
                logger.info("--- No Trade Executed ---")

        except Exception as e:
            logger.error(f"CRITICAL ERROR in main loop: {e}", exc_info=True)
            logger.info("Restarting loop in 60 seconds...")
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Bot shutting down by user request.")
            break