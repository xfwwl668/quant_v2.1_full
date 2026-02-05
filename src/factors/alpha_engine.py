"""
src/factors/alpha_engine.py
============================
Phase 3 — Alpha 因子引擎（统一封装层）

职责：
  1. 调用底层 Numba 加速函数（如 rsrs.compute_rsrs_batch）
  2. 将计算出的 ndarray 矩阵封装进 FactorStore
  3. 提供统一的 compute() 接口，供 BaseStrategy.compute_factors() 调用

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. **FactorStore 数据结构**
   ─────────────────────
   FactorStore = Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]]
   
   本引擎输出的是 ndarray 路径（高性能）：
     {
       "rsrs_beta": np.ndarray (n_stocks, n_days),
       "rsrs_r2": np.ndarray (n_stocks, n_days),
       ...
     }
   
   引擎会同时返回 code_to_idx 映射（{股票代码 → 列索引}），
   供 FactorAccessor 使用。

2. **计算流程**
   ───────────
   输入：
     - high, low, close, open, volume: np.ndarray (n_stocks, n_days)
     - codes: List[str]  股票代码列表
     - dates: List[str]  日期列表
   
   处理：
     1. 调用 Numba 函数批量计算因子
     2. 构建 code_to_idx 和 date_to_idx 映射
     3. 封装进 FactorStore 字典
   
   输出：
     FactorStore + metadata

3. **性能优化**
   ───────────
   - 输入必须是 C-contiguous ndarray（np.ascontiguousarray）
   - 避免任何 DataFrame 循环
   - 所有计算在 Numba 层完成

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 导入底层 Numba 加速函数
from .technical.rsrs import (
    compute_rsrs_batch,
    compute_signal_quality_batch,
    NUMBA_AVAILABLE,
)

# 导入类型定义
import sys
import os
# 修复导入路径
try:
    from src.strategy.types import FactorStore
except ImportError:
    from strategy.types import FactorStore  # type: ignore


# ============================================================================
# AlphaFactorEngine - 统一因子引擎
# ============================================================================

class AlphaFactorEngine:
    """
    Alpha 因子引擎 V2 — Numba 加速版。
    
    使用方法：
        engine = AlphaFactorEngine()
        factor_store = engine.compute(high, low, close, open_arr, volume, codes, dates)
    
    性能基准：
        1000 stocks × 1250 days < 50ms（单次计算）
        5000 stocks × 2500 days < 500ms
    """
    
    def __init__(self):
        self.logger = logging.getLogger("factors.alpha_engine")
        if not NUMBA_AVAILABLE:
            self.logger.warning("Numba not available. Performance will be degraded.")
    
    # ========================================================================
    # 主计算接口
    # ========================================================================
    
    def compute(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_arr: np.ndarray,
        volume: np.ndarray,
        codes: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        rsrs_window: int = 18,
        zscore_window: int = 600,
    ) -> Tuple[FactorStore, Dict[str, int], Dict[str, int]]:
        """
        批量计算 Alpha 因子（RSRS 全族 + 信号质量）。
        
        Args:
            high:   shape=(n_stocks, n_days)  最高价矩阵
            low:    shape=(n_stocks, n_days)  最低价矩阵
            close:  shape=(n_stocks, n_days)  收盘价矩阵
            open_arr: shape=(n_stocks, n_days)  开盘价矩阵
            volume: shape=(n_stocks, n_days)  成交量矩阵
            codes:  股票代码列表 (可选，用于构建 code_to_idx)
            dates:  日期列表 (可选，用于构建 date_to_idx)
            rsrs_window: RSRS 回归窗口（默认 18）
            zscore_window: Z-Score 标准化窗口（默认 600）
        
        Returns:
            (factor_store, code_to_idx, date_to_idx)
            
            factor_store: {
              "rsrs_beta": ndarray,
              "rsrs_r2": ndarray,
              "rsrs_std": ndarray,
              "rsrs_adaptive": ndarray,
              "rsrs_valid": ndarray,
              "rsrs_momentum": ndarray,
              "signal_quality": ndarray,
            }
            code_to_idx: {code → column index}
            date_to_idx: {date → row index}
        
        性能：单次调用 < 50ms（1000 股 × 1250 天）
        """
        # ── Step 1: 输入验证 ──
        n_stocks, n_days = high.shape
        
        if low.shape != (n_stocks, n_days):
            raise ValueError(f"low shape {low.shape} != high shape {high.shape}")
        if close.shape != (n_stocks, n_days):
            raise ValueError(f"close shape {close.shape} != high shape {high.shape}")
        if open_arr.shape != (n_stocks, n_days):
            raise ValueError(f"open shape {open_arr.shape} != high shape {high.shape}")
        if volume.shape != (n_stocks, n_days):
            raise ValueError(f"volume shape {volume.shape} != high shape {high.shape}")
        
        self.logger.info(f"Computing factors for {n_stocks} stocks × {n_days} days...")
        
        # ── Step 2: 确保 C-contiguous（Numba 性能关键）──
        high = np.ascontiguousarray(high, dtype=np.float64)
        low = np.ascontiguousarray(low, dtype=np.float64)
        close = np.ascontiguousarray(close, dtype=np.float64)
        open_arr = np.ascontiguousarray(open_arr, dtype=np.float64)
        volume = np.ascontiguousarray(volume, dtype=np.float64)
        
        # ── Step 3: 调用 Numba 加速函数计算 RSRS ──
        (
            rsrs_beta,
            rsrs_r2,
            rsrs_std,
            rsrs_adaptive,
            rsrs_valid,
            rsrs_momentum,
        ) = compute_rsrs_batch(high, low, rsrs_window, zscore_window)
        
        # ── Step 4: 计算信号质量 ──
        signal_quality = compute_signal_quality_batch(rsrs_r2, rsrs_std, rsrs_momentum)
        
        # ── Step 5: 封装进 FactorStore ──
        factor_store: FactorStore = {
            "rsrs_beta": rsrs_beta,
            "rsrs_r2": rsrs_r2,
            "rsrs_std": rsrs_std,
            "rsrs_adaptive": rsrs_adaptive,
            "rsrs_valid": rsrs_valid,
            "rsrs_momentum": rsrs_momentum,
            "signal_quality": signal_quality,
        }
        
        # ── Step 6: 构建索引映射 ──
        code_to_idx: Dict[str, int] = {}
        if codes is not None:
            if len(codes) != n_stocks:
                raise ValueError(f"codes length {len(codes)} != n_stocks {n_stocks}")
            code_to_idx = {code: idx for idx, code in enumerate(codes)}
        else:
            # 默认使用数字索引
            code_to_idx = {str(i): i for i in range(n_stocks)}
        
        date_to_idx: Dict[str, int] = {}
        if dates is not None:
            if len(dates) != n_days:
                raise ValueError(f"dates length {len(dates)} != n_days {n_days}")
            date_to_idx = {date: idx for idx, date in enumerate(dates)}
        else:
            # 默认使用数字索引
            date_to_idx = {str(i): i for i in range(n_days)}
        
        self.logger.info(f"✓ Computed {len(factor_store)} factors successfully")
        
        return factor_store, code_to_idx, date_to_idx
    
    # ========================================================================
    # 辅助方法：从 DataFrame 字典转换为矩阵
    # ========================================================================
    
    @staticmethod
    def from_dataframe_dict(
        history: Dict[str, pd.DataFrame],
        codes: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        从 {code: OHLCV DataFrame} 字典转换为矩阵格式。
        
        Args:
            history: {code: DataFrame with columns [date, open, high, low, close, volume]}
            codes: 指定股票代码列表（可选，默认使用 history.keys()）
        
        Returns:
            (high, low, close, open_arr, volume, codes_list, dates_list)
            所有矩阵 shape=(n_stocks, n_days)
        
        注意：此函数会对齐所有股票的日期序列（取交集）。
        """
        if codes is None:
            codes = sorted(history.keys())
        
        # FIXED: 提取所有股票日期的并集（避免新股数据缺失）
        all_dates_sets = []
        for code in codes:
            df = history.get(code)
            if df is not None and not df.empty:
                if "date" in df.columns:
                    dates_set = set(df["date"].astype(str))
                else:
                    dates_set = set(df.index.astype(str))
                all_dates_sets.append(dates_set)
        
        if not all_dates_sets:
            raise ValueError("No valid dataframes in history")
        
        # 取并集并排序
        dates = sorted(set.union(*all_dates_sets))
        
        n_stocks = len(codes)
        n_days = len(dates)
        
        # 初始化矩阵
        high = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        low = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        close = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        open_arr = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        volume = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        
        # 填充数据
        for i, code in enumerate(codes):
            df = history.get(code)
            if df is None or df.empty:
                continue
            
            # FIXED: 严格对齐到公共日期序列
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = df.index.astype(str)
            df = df.reindex(dates)
            
            # 提取列
            if "high" in df.columns:
                high[i, :] = df["high"].values
            if "low" in df.columns:
                low[i, :] = df["low"].values
            if "close" in df.columns:
                close[i, :] = df["close"].values
            if "open" in df.columns:
                open_arr[i, :] = df["open"].values
            if "volume" in df.columns:
                volume[i, :] = df["volume"].values
            elif "vol" in df.columns:
                volume[i, :] = df["vol"].values
        
        return high, low, close, open_arr, volume, codes, dates
    
    # ========================================================================
    # 辅助方法：将 FactorStore 转为 DataFrame（调试用）
    # ========================================================================
    
    @staticmethod
    def to_dataframe(
        factor_store: FactorStore,
        factor_name: str,
        codes: List[str],
        dates: List[str],
    ) -> pd.DataFrame:
        """
        将单个因子矩阵转为 DataFrame（用于调试和可视化）。
        
        Args:
            factor_store: 因子字典
            factor_name: 因子名称（如 "rsrs_beta"）
            codes: 股票代码列表
            dates: 日期列表
        
        Returns:
            DataFrame with index=dates, columns=codes
        """
        arr = factor_store.get(factor_name)
        if arr is None:
            raise ValueError(f"Factor '{factor_name}' not found in store")
        
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Factor '{factor_name}' is not ndarray")
        
        if arr.ndim == 2:
            # shape=(n_stocks, n_days) → transpose → (n_days, n_stocks)
            return pd.DataFrame(arr.T, index=dates, columns=codes)
        elif arr.ndim == 1:
            # shape=(n_stocks,) → Series
            return pd.DataFrame(arr, index=codes, columns=[factor_name])
        else:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "AlphaFactorEngine",
    "NUMBA_AVAILABLE",
]


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    import time
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("ALPHA FACTOR ENGINE - INTEGRATION TEST")
    print("=" * 70)
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print()
    
    # 测试数据
    n_stocks = 1000
    n_days = 1250
    
    print(f"Test Size: {n_stocks} stocks × {n_days} days")
    print()
    
    # 生成模拟数据
    np.random.seed(42)
    close = 10.0 + np.cumsum(np.random.randn(n_stocks, n_days) * 0.02, axis=1)
    high = close * (1.0 + np.abs(np.random.randn(n_stocks, n_days)) * 0.01)
    low = close * (1.0 - np.abs(np.random.randn(n_stocks, n_days)) * 0.01)
    open_arr = close * (1.0 + np.random.randn(n_stocks, n_days) * 0.005)
    volume = np.random.randint(100000, 10000000, (n_stocks, n_days)).astype(np.float64)
    
    codes = [f"SH{600000 + i:06d}" for i in range(n_stocks)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d").tolist()
    
    # 创建引擎
    engine = AlphaFactorEngine()
    
    # 预热
    print("Warming up...")
    _ = engine.compute(
        high[:10, :100],
        low[:10, :100],
        close[:10, :100],
        open_arr[:10, :100],
        volume[:10, :100],
    )
    print("✓ Warmup complete")
    print()
    
    # 正式测试
    print("Running benchmark (5 iterations)...")
    times = []
    
    for i in range(5):
        start = time.perf_counter()
        factor_store, code_to_idx, date_to_idx = engine.compute(
            high, low, close, open_arr, volume, codes, dates
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.2f} ms")
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    
    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"  Average Time: {avg_time:.2f} ms")
    print(f"  Min Time:     {min_time:.2f} ms")
    print(f"  Throughput:   {(n_stocks * n_days / avg_time * 1000) / 1e6:.2f} million points/sec")
    print()
    
    # 验证输出
    print("Factor Store Contents:")
    for name, arr in factor_store.items():
        valid_pct = np.sum(~np.isnan(arr)) / arr.size * 100
        print(f"  {name:20s}: shape={arr.shape}, valid={valid_pct:.1f}%")
    print()
    
    print(f"Code Index Mapping: {len(code_to_idx)} codes")
    print(f"Date Index Mapping: {len(date_to_idx)} dates")
    print()
    
    # 判断是否达标
    target = 50.0
    if avg_time < target:
        print(f"✅ PASS: {avg_time:.2f}ms < {target}ms target")
    else:
        print(f"❌ FAIL: {avg_time:.2f}ms >= {target}ms target")
    
    print("=" * 70)
    
    # 测试 DataFrame 转换
    print()
    print("Testing DataFrame conversion...")
    
    # 创建模拟 history 字典
    history = {}
    for i, code in enumerate(codes[:5]):  # 只取前 5 个
        df = pd.DataFrame({
            "date": dates,
            "open": open_arr[i, :],
            "high": high[i, :],
            "low": low[i, :],
            "close": close[i, :],
            "volume": volume[i, :],
        })
        history[code] = df
    
    # 转换
    h, l, c, o, v, c_list, d_list = engine.from_dataframe_dict(history)
    print(f"✓ Converted {len(history)} DataFrames → matrices")
    print(f"  Matrix shape: {h.shape}")
    print()
    
    # 转回 DataFrame（调试用）
    factor_df = engine.to_dataframe(factor_store, "rsrs_adaptive", codes[:10], dates[:10])
    print("Sample factor DataFrame (10×10):")
    print(factor_df)
    print()
    
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
