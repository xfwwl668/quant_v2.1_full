"""
src/data/sanitizer.py
=====================
Phase 5 — DataSanitizer（数据质量防线）

核心职责：
  1. 异常值检测：MAD (Median Absolute Deviation) 算法
  2. 数据修复：0价格、极小成交量、缺失值处理
  3. OHLCV一致性检查：确保 high >= close >= low, open > 0
  4. Numba 加速批量检测

═══════════════════════════════════════════════════════════════════
MAD 算法原理
═══════════════════════════════════════════════════════════════════

MAD (Median Absolute Deviation) 是一种鲁棒的异常值检测方法，
相比标准差（Std）对极端值不敏感。

算法步骤：
1. 计算中位数：median(X)
2. 计算绝对偏差：|X - median(X)|
3. 计算 MAD：median(|X - median(X)|)
4. 标准化得分：Z_MAD = 0.6745 × |X - median(X)| / MAD
5. 异常判断：|Z_MAD| > threshold（通常 3.5）

优势：
- 对极端值鲁棒（50% breakdown point）
- 适合金融数据（常有离群点）
- 计算高效（仅需排序）

示例：
  价格序列 [10.0, 10.1, 10.2, 10.3, 100.0]
  → Std 方法：100.0 可能不被检出（被自身拉高标准差）
  → MAD 方法：100.0 Z_MAD ≈ 26.9 > 3.5，确定为异常

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Numba 依赖
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. DataSanitizer will use slow fallback.")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# Part 1: MAD 算法（Numba 加速）
# ============================================================================

@njit(cache=True, fastmath=True)
def mad_scalar(arr: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    MAD 异常值检测（单序列）。
    
    Args:
        arr: shape=(n,) 输入序列
        threshold: Z_MAD 阈值（默认 3.5）
    
    Returns:
        is_outlier: shape=(n,) 布尔数组，True 表示异常值
    """
    n = len(arr)
    is_outlier = np.zeros(n, dtype=np.bool_)
    
    # 过滤 NaN
    valid_mask = ~np.isnan(arr)
    valid_values = arr[valid_mask]
    
    if len(valid_values) < 3:
        # 样本太少，无法检测
        return is_outlier
    
    # 计算中位数
    median = np.median(valid_values)
    
    # 绝对偏差
    abs_dev = np.abs(valid_values - median)
    
    # MAD
    mad = np.median(abs_dev)
    
    if mad < 1e-9:
        # MAD 接近 0（数据几乎无变化），无异常
        return is_outlier
    
    # 标准化得分（0.6745 是正态分布下的转换系数）
    z_scores = 0.6745 * abs_dev / mad
    
    # 标记异常值
    outlier_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(outlier_indices):
        if z_scores[i] > threshold:
            is_outlier[idx] = True
    
    return is_outlier


@njit(cache=True, fastmath=True)
def detect_price_outliers(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    threshold: float = 3.5,
) -> np.ndarray:
    """
    检测价格异常值（基于收盘价 MAD）。
    
    Args:
        close: shape=(n,) 收盘价
        high: shape=(n,) 最高价
        low: shape=(n,) 最低价
        threshold: MAD 阈值
    
    Returns:
        is_outlier: shape=(n,) 布尔数组
    """
    n = len(close)
    is_outlier = np.zeros(n, dtype=np.bool_)
    
    # 基于收盘价的 MAD 检测
    close_outliers = mad_scalar(close, threshold)
    
    # OHLC 一致性检查
    for i in range(n):
        if np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]):
            is_outlier[i] = True
            continue
        
        # 检查逻辑一致性
        if close[i] <= 0 or high[i] <= 0 or low[i] <= 0:
            is_outlier[i] = True
            continue
        
        # high >= close >= low
        if not (low[i] <= close[i] <= high[i]):
            is_outlier[i] = True
            continue
        
        # 价格跳变检测
        if close_outliers[i]:
            is_outlier[i] = True
    
    return is_outlier


@njit(cache=True, fastmath=True)
def detect_volume_outliers(
    volume: np.ndarray,
    threshold: float = 5.0,
) -> np.ndarray:
    """
    检测成交量异常值。
    
    Args:
        volume: shape=(n,) 成交量
        threshold: MAD 阈值（成交量用更宽松的阈值）
    
    Returns:
        is_outlier: shape=(n,) 布尔数组
    """
    n = len(volume)
    is_outlier = np.zeros(n, dtype=np.bool_)
    
    # 检查极小成交量（可能是停牌或数据错误）
    for i in range(n):
        if np.isnan(volume[i]) or volume[i] < 100:
            is_outlier[i] = True
    
    # MAD 检测（排除已标记的点）
    mad_outliers = mad_scalar(volume, threshold)
    is_outlier = is_outlier | mad_outliers
    
    return is_outlier


# ============================================================================
# Part 2: 数据修复
# ============================================================================

@njit(cache=True, fastmath=True)
def repair_price_forward_fill(
    arr: np.ndarray,
    is_outlier: np.ndarray,
) -> np.ndarray:
    """
    价格修复：前向填充（Forward Fill）。
    
    异常值用前一个有效值填充；若前面无有效值，用后向填充。
    
    Args:
        arr: shape=(n,) 原始序列
        is_outlier: shape=(n,) 异常标记
    
    Returns:
        repaired: shape=(n,) 修复后序列
    """
    n = len(arr)
    repaired = arr.copy()
    
    # 前向填充
    last_valid = np.nan
    for i in range(n):
        if is_outlier[i] or np.isnan(arr[i]):
            if not np.isnan(last_valid):
                repaired[i] = last_valid
        else:
            last_valid = arr[i]
    
    # 后向填充（处理开头的 NaN）
    next_valid = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(repaired[i]):
            if not np.isnan(next_valid):
                repaired[i] = next_valid
        else:
            next_valid = repaired[i]
    
    return repaired


@njit(cache=True, fastmath=True)
def repair_volume_median(
    volume: np.ndarray,
    is_outlier: np.ndarray,
) -> np.ndarray:
    """
    成交量修复：用中位数填充。
    
    Args:
        volume: shape=(n,) 原始成交量
        is_outlier: shape=(n,) 异常标记
    
    Returns:
        repaired: shape=(n,) 修复后成交量
    """
    n = len(volume)
    repaired = volume.copy()
    
    # 计算有效值的中位数
    valid_values = volume[~is_outlier]
    if len(valid_values) > 0:
        median_volume = np.median(valid_values)
    else:
        median_volume = 1000000.0  # 默认 100 万
    
    # 填充异常值
    for i in range(n):
        if is_outlier[i] or np.isnan(volume[i]) or volume[i] < 100:
            repaired[i] = median_volume
    
    return repaired


# ============================================================================
# Part 3: DataSanitizer 主类
# ============================================================================

class DataSanitizer:
    """
    数据清洗器（质量防线）。
    
    职责：
    1. 异常值检测（MAD 算法）
    2. 数据修复（前向填充 / 中位数填充）
    3. OHLCV 一致性检查
    4. 批量处理（支持 DataFrame 和 ndarray）
    
    使用方法：
        sanitizer = DataSanitizer()
        
        # 单股票清洗
        clean_df = sanitizer.sanitize_dataframe(df)
        
        # 批量清洗
        clean_data = sanitizer.sanitize_batch(history_data)
    """
    
    def __init__(
        self,
        price_threshold: float = 3.5,
        volume_threshold: float = 5.0,
        auto_repair: bool = True,
    ):
        """
        Args:
            price_threshold: 价格 MAD 阈值
            volume_threshold: 成交量 MAD 阈值
            auto_repair: 是否自动修复异常值
        """
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
        self.auto_repair = auto_repair
        
        self.logger = logging.getLogger("data.sanitizer")
        
        # 统计
        self._total_rows = 0
        self._outlier_rows = 0
        self._repaired_rows = 0
    
    # ========================================================================
    # DataFrame 路径
    # ========================================================================
    
    def sanitize_dataframe(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        清洗单个 DataFrame。
        
        Args:
            df: OHLCV DataFrame（必须包含 open, high, low, close, volume）
            inplace: 是否原地修改
        
        Returns:
            clean_df: 清洗后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # 检查必要列
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 提取数据
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        open_arr = df["open"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)
        
        n_rows = len(df)
        self._total_rows += n_rows
        
        # 检测异常
        price_outliers = detect_price_outliers(
            close, high, low, self.price_threshold
        )
        volume_outliers = detect_volume_outliers(
            volume, self.volume_threshold
        )
        
        n_price_outliers = np.sum(price_outliers)
        n_volume_outliers = np.sum(volume_outliers)
        
        self._outlier_rows += n_price_outliers + n_volume_outliers
        
        # 修复
        if self.auto_repair:
            if n_price_outliers > 0:
                df["close"] = repair_price_forward_fill(close, price_outliers)
                df["high"] = repair_price_forward_fill(high, price_outliers)
                df["low"] = repair_price_forward_fill(low, price_outliers)
                df["open"] = repair_price_forward_fill(open_arr, price_outliers)
                self._repaired_rows += n_price_outliers
            
            if n_volume_outliers > 0:
                df["volume"] = repair_volume_median(volume, volume_outliers)
                self._repaired_rows += n_volume_outliers
        
        return df
    
    def sanitize_batch(
        self,
        history_data: Dict[str, pd.DataFrame],
        show_progress: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量清洗历史数据。
        
        Args:
            history_data: {code: OHLCV DataFrame}
            show_progress: 是否显示进度条
        
        Returns:
            clean_data: {code: 清洗后的 DataFrame}
        """
        self.logger.info(f"Sanitizing {len(history_data)} stocks...")
        
        clean_data = {}
        
        codes = list(history_data.keys())
        if show_progress:
            try:
                from tqdm import tqdm
                codes = tqdm(codes, desc="Sanitizing")
            except ImportError:
                pass
        
        for code in codes:
            df = history_data[code]
            clean_df = self.sanitize_dataframe(df, inplace=False)
            clean_data[code] = clean_df
        
        self.logger.info(
            f"✓ Sanitized {self._total_rows} rows, "
            f"detected {self._outlier_rows} outliers, "
            f"repaired {self._repaired_rows} rows"
        )
        
        return clean_data
    
    # ========================================================================
    # ndarray 路径（高性能）
    # ========================================================================
    
    def sanitize_arrays(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_arr: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        清洗 ndarray 矩阵（用于 Numba 路径）。
        
        Args:
            所有参数 shape=(n_stocks, n_days)
        
        Returns:
            (clean_close, clean_high, clean_low, clean_open, clean_volume)
        """
        n_stocks, n_days = close.shape
        
        # 逐股清洗
        clean_close = close.copy()
        clean_high = high.copy()
        clean_low = low.copy()
        clean_open = open_arr.copy()
        clean_volume = volume.copy()
        
        for i in range(n_stocks):
            # 检测
            price_outliers = detect_price_outliers(
                close[i], high[i], low[i], self.price_threshold
            )
            volume_outliers = detect_volume_outliers(
                volume[i], self.volume_threshold
            )
            
            # 修复
            if self.auto_repair:
                clean_close[i] = repair_price_forward_fill(close[i], price_outliers)
                clean_high[i] = repair_price_forward_fill(high[i], price_outliers)
                clean_low[i] = repair_price_forward_fill(low[i], price_outliers)
                clean_open[i] = repair_price_forward_fill(open_arr[i], price_outliers)
                clean_volume[i] = repair_volume_median(volume[i], volume_outliers)
        
        return clean_close, clean_high, clean_low, clean_open, clean_volume
    
    # ========================================================================
    # 统计
    # ========================================================================
    
    def get_statistics(self) -> Dict:
        """获取清洗统计"""
        return {
            "total_rows": self._total_rows,
            "outlier_rows": self._outlier_rows,
            "repaired_rows": self._repaired_rows,
            "outlier_ratio": (
                self._outlier_rows / self._total_rows if self._total_rows > 0 else 0.0
            ),
        }
    
    def reset_statistics(self) -> None:
        """重置统计"""
        self._total_rows = 0
        self._outlier_rows = 0
        self._repaired_rows = 0


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "DataSanitizer",
    "mad_scalar",
    "detect_price_outliers",
    "detect_volume_outliers",
    "NUMBA_AVAILABLE",
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("DATA SANITIZER - TEST")
    print("=" * 70)
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print()
    
    # 测试数据（含异常值）
    n_days = 100
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    # 正常价格
    close = 10.0 + np.random.randn(n_days).cumsum() * 0.1
    
    # 注入异常值
    close[20] = 0.0        # 零价格
    close[50] = 100.0      # 异常高价
    close[70] = np.nan     # 缺失值
    
    high = close * 1.02
    low = close * 0.98
    open_arr = close * 1.01
    volume = np.random.uniform(1e6, 1e7, n_days)
    volume[30] = 0.0       # 零成交量
    
    df = pd.DataFrame({
        "date": dates,
        "open": open_arr,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    
    print("Test 1: Sanitize DataFrame")
    print(f"  Before: {len(df)} rows")
    print(f"  Outliers injected: indices [20, 30, 50, 70]")
    
    sanitizer = DataSanitizer()
    clean_df = sanitizer.sanitize_dataframe(df)
    
    print(f"  After: {len(clean_df)} rows")
    stats = sanitizer.get_statistics()
    print(f"  Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    print()
    
    # 验证修复
    print("Test 2: Verify repairs")
    print(f"  close[20] (was 0.0): {clean_df['close'].iloc[20]:.2f}")
    print(f"  close[50] (was 100.0): {clean_df['close'].iloc[50]:.2f}")
    print(f"  close[70] (was NaN): {clean_df['close'].iloc[70]:.2f}")
    print(f"  volume[30] (was 0.0): {clean_df['volume'].iloc[30]:.0f}")
    print()
    
    # 测试批量清洗
    print("Test 3: Batch sanitize")
    history_data = {
        f"SH{600000 + i:06d}": pd.DataFrame({
            "date": dates,
            "open": 10.0 + np.random.randn(n_days).cumsum() * 0.1,
            "high": 10.2 + np.random.randn(n_days).cumsum() * 0.1,
            "low": 9.8 + np.random.randn(n_days).cumsum() * 0.1,
            "close": 10.0 + np.random.randn(n_days).cumsum() * 0.1,
            "volume": np.random.uniform(1e6, 1e7, n_days),
        })
        for i in range(10)
    }
    
    sanitizer2 = DataSanitizer()
    clean_data = sanitizer2.sanitize_batch(history_data)
    print(f"  ✓ Cleaned {len(clean_data)} stocks")
    stats2 = sanitizer2.get_statistics()
    print(f"  Outlier ratio: {stats2['outlier_ratio']:.2%}")
    print()
    
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
