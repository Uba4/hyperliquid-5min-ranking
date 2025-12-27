"""
Hyperliquid Token Relative Strength Ranking System - 5-MINUTE TIMEFRAME
With Telegram Notifications and Performance Tracking
Tracks entry/exit prices, time in Top 6, and max movements for backtesting
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Tuple, Optional
import time

# ===================== CONFIGURATION =====================
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
VOLUME_THRESHOLD = 300000  # $300k 24h volume
RSI_PERIOD = 14
MA_PERIOD = 14
SIGNAL_LINE = 50
LOOKBACK_PERIODS = 200  # 200 five-minute candles (~16-17 hours of data)
RESULTS_FILE = "top6_history_5min.json"
LAST_TOP6_FILE = "last_top6_5min.json"
TRACKING_FILE = "token_tracking_5min.json"  # New: Performance tracking
MAX_HISTORY_ENTRIES = 288  # Keep 24 hours of 5-min snapshots

# Telegram configuration (loaded from environment variables)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# ===================== API FUNCTIONS =====================

def fetch_perpetuals_meta() -> List[str]:
    """Fetch all available perpetual tokens from Hyperliquid"""
    payload = {"type": "meta"}
    try:
        response = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        tokens = [asset['name'] for asset in data['universe']]
        print(f"✅ Fetched {len(tokens)} perpetual tokens")
        return tokens
    except Exception as e:
        print(f"❌ Error fetching perpetuals meta: {e}")
        return []

def fetch_asset_contexts() -> Dict:
    """Fetch 24h volume and other metrics for all tokens"""
    payload = {"type": "metaAndAssetCtxs"}
    try:
        response = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse volume data
        volume_data = {}
        for i, ctx in enumerate(data[1]):
            token_name = data[0]['universe'][i]['name']
            volume_24h = float(ctx.get('dayNtlVlm', 0))  # 24h notional volume
            volume_data[token_name] = volume_24h
        
        print(f"✅ Fetched volume data for {len(volume_data)} tokens")
        return volume_data
    except Exception as e:
        print(f"❌ Error fetching asset contexts: {e}")
        return {}

def fetch_candles_5min(token: str, periods: int = LOOKBACK_PERIODS) -> pd.DataFrame:
    """Fetch 5-minute OHLCV data for a token"""
    end_time = int(time.time() * 1000)
    start_time = end_time - (periods * 5 * 60 * 1000)  # periods × 5 min × 60 sec × 1000 ms
    
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": token,
            "interval": "5m",  # 5-minute candles
            "startTime": start_time,
            "endTime": end_time
        }
    }
    
    try:
        response = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
        response.raise_for_status()
        candles = response.json()
        
        if not candles:
            return pd.DataFrame()
        
        # Parse candle data
        df = pd.DataFrame(candles)
        df['close'] = df['c'].astype(float)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.sort_values('timestamp')
        
        return df[['timestamp', 'close']]
    except Exception as e:
        print(f"⚠️  Error fetching candles for {token}: {e}")
        return pd.DataFrame()

# ===================== TECHNICAL INDICATORS =====================

def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi_ma(rsi: pd.Series, period: int = MA_PERIOD) -> float:
    """Calculate moving average of RSI and return latest value"""
    rsi_ma = rsi.rolling(window=period).mean()
    return rsi_ma.iloc[-1] if not rsi_ma.empty else 50.0

# ===================== RATIO ANALYSIS =====================

def create_ratio_series(token1_data: pd.DataFrame, token2_data: pd.DataFrame) -> pd.Series:
    """Create ratio series from two token price series"""
    # Merge on timestamp
    merged = pd.merge(token1_data, token2_data, on='timestamp', suffixes=('_1', '_2'))
    ratio = merged['close_1'] / merged['close_2']
    return ratio

def analyze_ratio(token1: str, token2: str, data_cache: Dict) -> Tuple[str, str, float]:
    """
    Analyze a ratio pair and return winner, loser, and RSI-MA value
    Returns: (winner_token, loser_token, rsi_ma_value)
    """
    if token1 not in data_cache or token2 not in data_cache:
        return None, None, 50.0
    
    token1_data = data_cache[token1]
    token2_data = data_cache[token2]
    
    if token1_data.empty or token2_data.empty:
        return None, None, 50.0
    
    # Create ratio: token1/token2
    ratio_series = create_ratio_series(token1_data, token2_data)
    
    if len(ratio_series) < RSI_PERIOD + MA_PERIOD:
        return None, None, 50.0
    
    # Calculate RSI
    rsi = calculate_rsi(ratio_series, RSI_PERIOD)
    
    # Calculate RSI-MA
    rsi_ma = calculate_rsi_ma(rsi, MA_PERIOD)
    
    # Determine winner/loser based on RSI-MA vs signal line (50)
    if rsi_ma > SIGNAL_LINE:
        return token1, token2, rsi_ma  # token1 wins
    else:
        return token2, token1, rsi_ma  # token2 wins

# ===================== SCORING ENGINE =====================

def calculate_scores(tokens: List[str], data_cache: Dict) -> Dict[str, int]:
    """
    Calculate scores for all tokens based on ratio analysis
    Each win = +1, each loss = 0
    """
    scores = {token: 0 for token in tokens}
    total_comparisons = 0
    successful_comparisons = 0
    
    print(f"\n🔄 Analyzing {len(tokens) * (len(tokens) - 1)} ratio pairs...")
    
    # Analyze all possible pairs (including both directions)
    for i, token1 in enumerate(tokens):
        for token2 in tokens:
            if token1 == token2:
                continue
            
            total_comparisons += 1
            winner, loser, rsi_ma = analyze_ratio(token1, token2, data_cache)
            
            if winner and loser:
                scores[winner] += 1
                successful_comparisons += 1
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1}/{len(tokens)} tokens...")
    
    print(f"✅ Completed {successful_comparisons}/{total_comparisons} ratio analyses")
    return scores

# ===================== PERFORMANCE TRACKING =====================

def load_tracking_data() -> Dict:
    """Load token tracking data"""
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading tracking data: {e}")
    return {}

def save_tracking_data(tracking: Dict):
    """Save token tracking data"""
    try:
        with open(TRACKING_FILE, 'w') as f:
            json.dump(tracking, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving tracking data: {e}")

def update_tracking(current_top6: List[Tuple[str, int]], data_cache: Dict, tracking: Dict) -> Dict:
    """
    Update tracking data for tokens in Top 6
    Tracks entry, updates max/min, and handles exits
    """
    now = datetime.utcnow().isoformat()
    current_tokens = [token for token, score in current_top6]
    
    # Update existing tokens and track new entries
    for token, score in current_top6:
        if token in data_cache and not data_cache[token].empty:
            current_price = float(data_cache[token]['close'].iloc[-1])
            
            if token not in tracking:
                # New entry to Top 6
                tracking[token] = {
                    "entry_time": now,
                    "entry_price": current_price,
                    "current_price": current_price,
                    "max_price": current_price,
                    "min_price": current_price,
                    "in_top6": True
                }
                print(f"📊 Started tracking {token} @ ${current_price:.4g}")
            else:
                # Update existing tracking
                tracking[token]["current_price"] = current_price
                tracking[token]["max_price"] = max(tracking[token]["max_price"], current_price)
                tracking[token]["min_price"] = min(tracking[token]["min_price"], current_price)
                tracking[token]["in_top6"] = True
    
    # Mark tokens that left Top 6
    for token in list(tracking.keys()):
        if tracking[token].get("in_top6", False) and token not in current_tokens:
            tracking[token]["in_top6"] = False
            tracking[token]["exit_time"] = now
            if token in data_cache and not data_cache[token].empty:
                tracking[token]["exit_price"] = float(data_cache[token]['close'].iloc[-1])
            print(f"📊 {token} exited Top 6")
    
    return tracking

def calculate_performance_stats(token: str, tracking_data: Dict) -> Dict:
    """Calculate performance statistics for a token that exited Top 6"""
    if token not in tracking_data:
        return None
    
    data = tracking_data[token]
    entry_price = data.get("entry_price", 0)
    exit_price = data.get("exit_price", entry_price)
    max_price = data.get("max_price", entry_price)
    min_price = data.get("min_price", entry_price)
    entry_time = data.get("entry_time", "")
    exit_time = data.get("exit_time", "")
    
    if entry_price == 0:
        return None
    
    # Calculate percentages
    net_move_pct = ((exit_price - entry_price) / entry_price) * 100
    max_upward_pct = ((max_price - entry_price) / entry_price) * 100
    max_downward_pct = ((min_price - entry_price) / entry_price) * 100
    
    # Calculate time spent
    try:
        entry_dt = datetime.fromisoformat(entry_time)
        exit_dt = datetime.fromisoformat(exit_time)
        time_delta = exit_dt - entry_dt
        
        days = time_delta.days
        hours = time_delta.seconds // 3600
        minutes = (time_delta.seconds % 3600) // 60
        
        if days > 0:
            time_str = f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m"
    except:
        time_str = "Unknown"
    
    return {
        "entry_price": entry_price,
        "entry_time": entry_time,
        "exit_price": exit_price,
        "exit_time": exit_time,
        "net_move_pct": net_move_pct,
        "max_upward_pct": max_upward_pct,
        "max_upward_price": max_price,
        "max_downward_pct": max_downward_pct,
        "max_downward_price": min_price,
        "time_in_top6": time_str
    }

# ===================== TELEGRAM NOTIFICATIONS =====================

def send_telegram_message(message: str) -> bool:
    """Send a message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram not configured (missing bot token or chat ID)")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Telegram notification sent successfully")
        return True
    except Exception as e:
        print(f"❌ Error sending Telegram notification: {e}")
        return False

def load_last_top6() -> Optional[Dict]:
    """Load the previous Top 6 results for comparison"""
    if os.path.exists(LAST_TOP6_FILE):
        try:
            with open(LAST_TOP6_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading last Top 6: {e}")
    return None

def save_last_top6(top6: List[Tuple[str, int]]):
    """Save current Top 6 for next comparison"""
    data = {
        "tokens": [token for token, score in top6],
        "scores": [score for token, score in top6],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        with open(LAST_TOP6_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved current Top 6 to {LAST_TOP6_FILE}")
    except Exception as e:
        print(f"❌ Error saving last Top 6: {e}")

def detect_changes(previous: Optional[Dict], current: List[Tuple[str, int]]) -> Optional[Dict]:
    """
    Detect changes between previous and current Top 6
    Returns a dict with change details or None if no changes
    Only triggers on: new entries or dropouts (NOT position-only changes)
    """
    if not previous:
        return None  # First run, no comparison possible
    
    prev_tokens = previous.get('tokens', [])
    curr_tokens = [token for token, score in current]
    
    # Check if the COMPOSITION changed (ignore order)
    if set(prev_tokens) == set(curr_tokens):
        return None  # Same tokens (just different order) = no notification
    
    changes = {
        "new_entries": [],
        "dropped_out": []
    }
    
    # Detect new entries
    for token in curr_tokens:
        if token not in prev_tokens:
            changes["new_entries"].append(token)
    
    # Detect dropped tokens
    for token in prev_tokens:
        if token not in curr_tokens:
            changes["dropped_out"].append(token)
    
    return changes

def format_telegram_notification(changes: Dict, current: List[Tuple[str, int]], previous: Dict, data_cache: Dict, tracking: Dict) -> str:
    """Format changes into a clean notification with performance tracking"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    message = f"⚡ <b>5-MIN TOP 6 CHANGE</b>\n"
    message += f"⏰ Time: {timestamp}\n\n"
    
    # New entries with prices
    if changes.get("new_entries"):
        for token in changes["new_entries"]:
            if token in data_cache and not data_cache[token].empty:
                price = data_cache[token]['close'].iloc[-1]
                message += f"📈 <b>New Entry:</b> {token} (${price:,.4g})\n\n"
    
    # Dropped out with performance stats
    if changes.get("dropped_out"):
        for token in changes["dropped_out"]:
            stats = calculate_performance_stats(token, tracking)
            if stats:
                entry_time_short = stats['entry_time'][:16].replace('T', ' ')
                exit_time_short = stats['exit_time'][:16].replace('T', ' ')
                
                message += f"📉 <b>Dropped Out:</b> {token}\n"
                message += f"   Entry: ${stats['entry_price']:,.4g} @ {entry_time_short}\n"
                message += f"   Exit: ${stats['exit_price']:,.4g} @ {exit_time_short}\n"
                message += f"   Net Move: {stats['net_move_pct']:+.2f}%\n"
                message += f"   Time in Top 6: {stats['time_in_top6']}\n"
                message += f"   Max Upward: +{stats['max_upward_pct']:.2f}% (${stats['max_upward_price']:,.4g})\n"
                message += f"   Max Downward: {stats['max_downward_pct']:+.2f}% (${stats['max_downward_price']:,.4g})\n\n"
    
    # Previous and New Top 6
    prev_tokens = ', '.join(previous.get('tokens', []))
    curr_tokens = ', '.join([token for token, score in current])
    
    message += f"<b>Previous TOP 6:</b> {prev_tokens}\n"
    message += f"<b>New TOP 6:</b> {curr_tokens}"
    
    return message

# ===================== RESULTS MANAGEMENT =====================

def save_results(top6: List[Tuple[str, int]]):
    """Save top 6 results to JSON file with timestamp-based tracking"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    
    # Load existing history
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = {}
    
    # Add current results with timestamp
    history[timestamp] = {
        "tokens": [{"token": token, "score": score} for token, score in top6],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Keep only last MAX_HISTORY_ENTRIES (24 hours of 5-min data)
    sorted_timestamps = sorted(history.keys(), reverse=True)
    if len(sorted_timestamps) > MAX_HISTORY_ENTRIES:
        for old_timestamp in sorted_timestamps[MAX_HISTORY_ENTRIES:]:
            del history[old_timestamp]
    
    # Save to file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Results saved to {RESULTS_FILE}")

def display_results(top6: List[Tuple[str, int]], volume_data: Dict):
    """Display results in a clean, readable format"""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    print("\n" + "=" * 60)
    print(f"⚡ TOP 6 TOKENS (5-MIN) - {now}")
    print("=" * 60)
    
    for rank, (token, score) in enumerate(top6, 1):
        volume = volume_data.get(token, 0)
        print(f"#{rank}  {token:8s}  |  Score: {score:3d}  |  24h Volume: ${volume:,.0f}")
    
    print("=" * 60)

def display_history():
    """Display recent historical results if available"""
    if not os.path.exists(RESULTS_FILE):
        return
    
    with open(RESULTS_FILE, 'r') as f:
        history = json.load(f)
    
    if not history:
        return
    
    print("\n" + "=" * 60)
    print("📊 RECENT TOP 6 (Last 10 snapshots)")
    print("=" * 60)
    
    # Show last 10 entries
    sorted_timestamps = sorted(history.keys(), reverse=True)[:10]
    
    for ts in sorted_timestamps:
        tokens = [item['token'] for item in history[ts]['tokens']]
        print(f"{ts}: {', '.join(tokens)}")
    
    print("=" * 60)

# ===================== MAIN EXECUTION =====================

def main():
    print("=" * 60)
    print("⚡ HYPERLIQUID 5-MINUTE RANKING SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Timeframe: 5-minute candles")
    print(f"Volume threshold: ${VOLUME_THRESHOLD:,}")
    print(f"RSI period: {RSI_PERIOD}, MA period: {MA_PERIOD}")
    
    # Load previous Top 6 for comparison
    previous_top6 = load_last_top6()
    
    # Load tracking data
    tracking = load_tracking_data()
    
    # Step 1: Fetch all perpetuals
    print("\n📡 Step 1: Fetching perpetual tokens...")
    all_tokens = fetch_perpetuals_meta()
    
    if not all_tokens:
        print("❌ Failed to fetch tokens. Exiting.")
        return
    
    # Step 2: Fetch volume data and filter
    print("\n📡 Step 2: Fetching 24h volume data...")
    volume_data = fetch_asset_contexts()
    
    if not volume_data:
        print("❌ Failed to fetch volume data. Exiting.")
        return
    
    # Filter tokens by volume
    filtered_tokens = [
        token for token in all_tokens 
        if volume_data.get(token, 0) > VOLUME_THRESHOLD
    ]
    
    print(f"✅ {len(filtered_tokens)} tokens passed volume filter (> ${VOLUME_THRESHOLD:,})")
    print(f"   Tokens: {', '.join(filtered_tokens[:10])}{'...' if len(filtered_tokens) > 10 else ''}")
    
    if len(filtered_tokens) < 2:
        print("❌ Not enough tokens to analyze. Exiting.")
        return
    
    # Step 3: Fetch historical data for all filtered tokens
    print(f"\n📡 Step 3: Fetching {LOOKBACK_PERIODS} five-minute candles (~{LOOKBACK_PERIODS * 5 / 60:.1f} hours of data)...")
    data_cache = {}
    
    for i, token in enumerate(filtered_tokens):
        df = fetch_candles_5min(token, LOOKBACK_PERIODS)
        if not df.empty:
            data_cache[token] = df
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Fetched data for {i + 1}/{len(filtered_tokens)} tokens...")
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"✅ Successfully fetched data for {len(data_cache)} tokens")
    
    # Step 4: Calculate scores
    print("\n🔬 Step 4: Running ratio analysis and scoring...")
    scores = calculate_scores(list(data_cache.keys()), data_cache)
    
    # Step 5: Get top 6
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top6 = sorted_scores[:6]
    
    # Step 6: Display results
    display_results(top6, volume_data)
    
    # Step 7: Update tracking data
    print("\n📊 Step 7: Updating performance tracking...")
    tracking = update_tracking(top6, data_cache, tracking)
    save_tracking_data(tracking)
    
    # Step 8: Detect changes and send notification
    print("\n🔍 Step 8: Detecting changes...")
    
    # Debug: Show what we're comparing
    if previous_top6:
        prev_tokens = previous_top6.get('tokens', [])
        curr_tokens = [token for token, score in top6]
        print(f"📋 Previous Top 6: {', '.join(prev_tokens)}")
        print(f"📋 Current Top 6:  {', '.join(curr_tokens)}")
        
        if set(prev_tokens) != set(curr_tokens):
            print("🔍 Token composition changed! → Will notify")
        else:
            print("✅ Same tokens (just different order) → No notification")
    else:
        print("📋 First run - no previous data to compare")
    
    changes = detect_changes(previous_top6, top6)
    
    if changes:
        print("🔔 Changes detected! Sending Telegram notification...")
        message = format_telegram_notification(changes, top6, previous_top6, data_cache, tracking)
        send_telegram_message(message)
        
        # Clean up tracking for tokens that exited
        for token in changes.get("dropped_out", []):
            if token in tracking:
                del tracking[token]
                print(f"🗑️  Removed {token} from tracking (exited Top 6)")
        save_tracking_data(tracking)
    else:
        print("✅ No meaningful changes detected. No notification sent.")
    
    # Step 9: Save results
    save_results(top6)
    save_last_top6(top6)
    display_history()
    
    print(f"\n✅ Analysis completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

if __name__ == "__main__":
    main()
