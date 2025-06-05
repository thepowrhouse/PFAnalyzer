import numpy as np
import pandas as pd
from scipy.signal import find_peaks, argrelextrema
import matplotlib.pyplot as plt


# Helper function to calculate swing amplitude
def calculate_swing_amplitude(highs, lows):
    return [highs[i] - lows[i] for i in range(len(highs))]


# ========================
# HEAD & SHOULDERS PATTERN
# ========================
def detect_head_and_shoulders(prices, volumes=None):
    peaks, _ = find_peaks(prices)
    valleys, _ = find_peaks(-prices)

    if len(peaks) < 3 or len(valleys) < 2:
        return False

    for i in range(len(peaks) - 2):
        l, h, r = peaks[i], peaks[i + 1], peaks[i + 2]

        # Shoulder symmetry (within 3%)
        shoulder_diff = abs(prices.iloc[l] - prices.iloc[r]) / max(prices.iloc[l], prices.iloc[r])
        if shoulder_diff > 0.03:
            continue

        # Head prominence (at least 10% higher)
        if prices.iloc[h] < prices.iloc[l] * 1.10 or prices.iloc[h] < prices.iloc[r] * 1.10:
            continue

        # Peak spacing (3-20 periods)
        if not (3 < h - l < 20 and 3 < r - h < 20):
            continue

        # Find valleys between peaks
        left_valleys = [v for v in valleys if l < v < h]
        right_valleys = [v for v in valleys if h < v < r]

        if not left_valleys or not right_valleys:
            continue

        valley1 = min(left_valleys, key=lambda v: prices.iloc[v])
        valley2 = min(right_valleys, key=lambda v: prices.iloc[v])

        # Neckline validation
        neckline_slope = (prices.iloc[valley2] - prices.iloc[valley1]) / (valley2 - valley1)
        neckline_func = lambda idx: prices.iloc[valley1] + neckline_slope * (idx - valley1)

        # Volume surge at head (if volume data available)
        if volumes is not None:
            head_volume = volumes.iloc[h]
            avg_volume = volumes.iloc[l:r + 1].mean()
            if head_volume < avg_volume * 1.5:  # Require 50% volume surge
                continue

        # Neckline breakout confirmation
        breakout_window = prices.iloc[r + 1:r + 10]
        if not any(prices.iloc[r + 1 + i] < neckline_func(r + 1 + i) for i in range(len(breakout_window))):
            continue

        return True

    return False


# =================================
# INVERTED HEAD & SHOULDERS PATTERN
# =================================
def detect_inverted_head_and_shoulders(prices, volumes=None):
    valleys, _ = find_peaks(-prices)
    peaks, _ = find_peaks(prices)

    if len(valleys) < 3 or len(peaks) < 2:
        return False

    for i in range(len(valleys) - 2):
        l, h, r = valleys[i], valleys[i + 1], valleys[i + 2]

        # Shoulder symmetry
        shoulder_diff = abs(prices.iloc[l] - prices.iloc[r]) / max(prices.iloc[l], prices.iloc[r])
        if shoulder_diff > 0.03:
            continue

        # Head prominence (at least 10% lower)
        if prices.iloc[h] > prices.iloc[l] * 0.90 or prices.iloc[h] > prices.iloc[r] * 0.90:
            continue

        # Valley spacing (3-20 periods)
        if not (3 < h - l < 20 and 3 < r - h < 20):
            continue

        # Find peaks between valleys
        left_peaks = [p for p in peaks if l < p < h]
        right_peaks = [p for p in peaks if h < p < r]

        if not left_peaks or not right_peaks:
            continue

        peak1 = max(left_peaks, key=lambda p: prices.iloc[p])
        peak2 = max(right_peaks, key=lambda p: prices.iloc[p])

        # Neckline validation
        neckline_slope = (prices.iloc[peak2] - prices.iloc[peak1]) / (peak2 - peak1)
        neckline_func = lambda idx: prices.iloc[peak1] + neckline_slope * (idx - peak1)

        # Volume surge at head
        if volumes is not None:
            head_volume = volumes.iloc[h]
            avg_volume = volumes.iloc[l:r + 1].mean()
            if head_volume < avg_volume * 1.5:
                continue

        # Neckline breakout confirmation
        breakout_window = prices.iloc[r + 1:r + 10]
        if not any(prices.iloc[idx] > neckline_func(idx) for idx in breakout_window.index):
            continue

        return True

    return False


# ================
# DOUBLE TOP/BOTTOM
# ================
def validate_double_pattern(extrema, prices, is_top=True, min_periods=10, symmetry_threshold=0.4):
    for i in range(len(extrema) - 1):
        e1, e2 = extrema[i], extrema[i + 1]

        # Price similarity (within 3%)
        price_diff = abs(prices.iloc[e1] - prices.iloc[e2]) / prices.iloc[e1]
        if price_diff > 0.03:
            continue

        # Minimum duration requirement
        if abs(e2 - e1) < min_periods:
            continue

        # Time symmetry validation
        midpoint = (e1 + e2) // 2
        left_duration = midpoint - e1
        right_duration = e2 - midpoint
        symmetry_ratio = min(left_duration, right_duration) / max(left_duration, right_duration)
        if symmetry_ratio < symmetry_threshold:
            continue

        return True

    return False


def detect_double_top(prices):
    peaks, _ = find_peaks(prices)
    if len(peaks) < 2:
        return False
    return validate_double_pattern(peaks, prices, is_top=True)


def detect_double_bottom(prices):
    valleys, _ = find_peaks(-prices)
    if len(valleys) < 2:
        return False
    return validate_double_pattern(valleys, prices, is_top=False)


# =================
# TRIPLE TOP/BOTTOM
# =================
def detect_triple_top(prices):
    peaks, _ = find_peaks(prices)
    if len(peaks) < 3:
        return False

    for i in range(len(peaks) - 2):
        p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]

        # Price similarity
        if (abs(prices.iloc[p1] - prices.iloc[p2]) / prices.iloc[p1] > 0.03 or
                abs(prices.iloc[p2] - prices.iloc[p3]) / prices.iloc[p2] > 0.03):
            continue

        # Time symmetry validation
        duration1 = p2 - p1
        duration2 = p3 - p2
        symmetry_ratio = min(duration1, duration2) / max(duration1, duration2)
        if symmetry_ratio < 0.4:  # 40% symmetry threshold
            continue

        # Minimum duration requirement
        if duration1 < 10 or duration2 < 10:
            continue

        return True

    return False


def detect_triple_bottom(prices):
    valleys, _ = find_peaks(-prices)
    if len(valleys) < 3:
        return False

    for i in range(len(valleys) - 2):
        v1, v2, v3 = valleys[i], valleys[i + 1], valleys[i + 2]

        # Price similarity
        if (abs(prices.iloc[v1] - prices.iloc[v2]) / prices.iloc[v1] > 0.03 or
                abs(prices.iloc[v2] - prices.iloc[v3]) / prices.iloc[v2] > 0.03):
            continue

        # Time symmetry validation
        duration1 = v2 - v1
        duration2 = v3 - v2
        symmetry_ratio = min(duration1, duration2) / max(duration1, duration2)
        if symmetry_ratio < 0.4:
            continue

        # Minimum duration requirement
        if duration1 < 10 or duration2 < 10:
            continue

        return True

    return False


# =============
# CUP AND HANDLE
# =============
def detect_cup_and_handle(prices, volumes=None, window=30, handle_window=10):
    for i in range(window, len(prices) - handle_window - 1):
        left_max = prices.iloc[i - window:i].max()
        cup_min = prices.iloc[i - window:i + 1].min()
        right_max = prices.iloc[i + 1:i + 1 + window].max()

        # Cup depth validation (30-50% retracement)
        retracement = (left_max - cup_min) / left_max
        if not (0.3 <= retracement <= 0.5):
            continue

        # Handle should form in upper half of cup
        cup_midpoint = (left_max + cup_min) / 2
        handle_prices = prices.iloc[i + 1:i + 1 + handle_window]
        if handle_prices.min() < cup_midpoint:
            continue

        # Volume dry-up in handle (if volume available)
        if volumes is not None:
            cup_volume = volumes.iloc[i - window:i + 1].mean()
            handle_volume = volumes.iloc[i + 1:i + 1 + handle_window].mean()
            if handle_volume > cup_volume * 0.8:  # Require at least 20% volume reduction
                continue

        return True

    return False


# =================
# TRIANGLE PATTERNS
# =================
def validate_triangle(prices, min_touch_points=5):
    prices = pd.Series(prices).reset_index(drop=True)
    highs = prices.rolling(window=5, min_periods=1).max().reset_index(drop=True)
    lows = prices.rolling(window=5, min_periods=1).min().reset_index(drop=True)

    peaks, _ = find_peaks(prices)
    valleys, _ = find_peaks(-prices)

    # Only use peaks/valleys within the valid range of highs/lows
    resistance_touches = [p for p in peaks if p < len(highs) and abs(prices.iloc[p] - highs.iloc[p]) < 0.01 * prices.mean()]
    support_touches = [v for v in valleys if v < len(lows) and abs(prices.iloc[v] - lows.iloc[v]) < 0.01 * prices.mean()]

    if len(resistance_touches) + len(support_touches) < min_touch_points:
        return False

    if len(peaks) < 3 or len(valleys) < 3:
        return False

    swing_amps = []
    for i in range(min(len(peaks), len(valleys))):
        if peaks[i] > valleys[i]:
            swing_amps.append(prices.iloc[peaks[i]] - prices.iloc[valleys[i]])

    if len(swing_amps) < 3:
        return False

    contraction_ratio = swing_amps[-1] / swing_amps[0]
    if contraction_ratio > 0.8:
        return False

    return len(support_touches) >= 2


def detect_symmetrical_triangle(prices):
    if not validate_triangle(prices):
        return False

    # Breakout direction confirmation
    if len(prices) < 30:
        return False

    # Look for breakout in either direction
    high = prices.max()
    low = prices.min()
    mid = (high + low) / 2
    last_price = prices.iloc[-1]

    # Breakout confirmation
    if last_price > high * 1.02 or last_price < low * 0.98:
        return True

    return False


# ===================
# DETECT ALL PATTERNS
# ===================

PATTERN_RECOMMENDATIONS = {
    "Cup and Handle": "buy",
    "Head & Shoulders": "sell",
    "Inverted Head & Shoulders": "buy",
    "Double Top": "sell",
    "Double Bottom": "buy",
    "Triple Top": "sell",
    "Triple Bottom": "buy",
    "Symmetrical Triangle": "hold",  # or "buy"/"sell" if you want to infer direction
}

def detect_chart_pattern(prices, volumes=None):
    patterns = [
        ("Cup and Handle", lambda: detect_cup_and_handle(prices, volumes)),
        ("Head & Shoulders", lambda: detect_head_and_shoulders(prices, volumes)),
        ("Inverted Head & Shoulders", lambda: detect_inverted_head_and_shoulders(prices, volumes)),
        ("Double Top", lambda: detect_double_top(prices)),
        ("Double Bottom", lambda: detect_double_bottom(prices)),
        ("Triple Top", lambda: detect_triple_top(prices)),
        ("Triple Bottom", lambda: detect_triple_bottom(prices)),
        ("Symmetrical Triangle", lambda: detect_symmetrical_triangle(prices)),
    ]

    for name, detector in patterns:
        if detector():
            recommendation = PATTERN_RECOMMENDATIONS.get(name, "hold")
            return name, recommendation

    return "No clear pattern", "hold"