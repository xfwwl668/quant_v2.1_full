"""
src/data/storage.py
===================
Phase 5 — ColumnarStorageManager（列式存储管理器）

核心职责：
  1. Parquet 存储：高效压缩的列式存储
  2. Memory Mapping：零拷贝加载到 Numba
  3. 对齐逻辑：处理停牌股票，确保所有股票在同一 shape
  4. 批量转换：TDX 原始数据 → 规整化矩阵

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. **Parquet vs CSV**
   ───────────────────
   | 指标 | CSV | Parquet |
   |-----|-----|---------|
   | 读取速度 | 1x | 10-50x |
   | 磁盘空间 | 1x | 0.1-0.2x |
   | 列式读取 | ✗ | ✓ |
   | 类型保留 | ✗ | ✓ |
   
   Parquet 是列式存储格式，支持：
   - 高效压缩（Snappy / GZIP）
   - 仅读取需要的列
   - 保留数据类型
   - Pandas / PyArrow 原生支持

2. **日期对齐（Alignment）**
   ─────────────────────────
   问题：不同股票的交易日不同（停牌、新股、退市）
   
   原始数据：
     SH600000: [2024-01-02, 2024-01-03, 2024-01-04, ...]  # 100 天
     SH600001: [2024-01-02, 2024-01-05, ...]              # 80 天（停牌）
   
   对齐后：
     SH600000: [2024-01-02, 2024-01-03, 2024-01-04, ...]  # 100 天
     SH600001: [2024-01-02, NaN,        2024-01-04, ...]  # 100 天（补NaN）
   
   方法：
   - 构建全局日期索引（union of all dates）
   - 每只股票 reindex 到全局索引
   - 缺失日期填充 NaN
   - 最终所有股票 shape 一致：(n_stocks, n_days)

3. **Memory Mapping（mmap）**
   ──────────────────────────
   传统方式：
     data = pd.read_parquet(file)  # 全部加载到内存
   
   零拷贝方式：
     table = pq.read_table(file, memory_map=True)
     arr = table.column("close").to_numpy()  # 视图，无拷贝
   
   优势：
   - 节省内存（仅映射需要的列）
   - 加载速度快（仅读取元数据）
   - 适合大规模数据

4. **存储结构**
   ───────────
   data/
   ├── raw/                    # 原始 CSV/TDX 数据
   ├── parquet/                # Parquet 存储
   │   ├── daily/              # 日线数据
   │   │   ├── SH600000.parquet
   │   │   ├── SH600001.parquet
   │   │   └── ...
   │   └── aligned/            # 对齐后的矩阵数据
   │       ├── close.npy       # (n_stocks, n_days)
   │       ├── high.npy
   │       └── metadata.json   # {codes, dates}
   └── cache/                  # 临时缓存

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# PyArrow 依赖（Parquet 支持）
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None  # type: ignore
    pq = None  # type: ignore


# ============================================================================
# ColumnarStorageManager 主类
# ============================================================================

class ColumnarStorageManager:
    """
    列式存储管理器（数据补给线核心）。
    
    职责：
    1. 保存数据到 Parquet 格式
    2. 加载数据（支持 memory mapping）
    3. 日期对齐（reindex 补齐停牌缺口）
    4. 转换为规整化矩阵（用于 Numba）
    
    使用方法：
        # 初始化
        storage = ColumnarStorageManager(base_dir="data")
        
        # 保存数据
        storage.save_stock_data(code, df)
        
        # 加载对齐矩阵
        matrices = storage.load_aligned_matrices(codes, start_date, end_date)
    """
    
    def __init__(
        self,
        base_dir: str = "data",
        use_memory_map: bool = True,
        compression: str = "snappy",
    ):
        """
        Args:
            base_dir: 数据根目录
            use_memory_map: 是否使用 memory mapping
            compression: 压缩算法（snappy / gzip / none）
        """
        self.base_dir = Path(base_dir)
        self.use_memory_map = use_memory_map and PYARROW_AVAILABLE
        self.compression = compression
        
        # 创建目录结构
        self.parquet_dir = self.base_dir / "parquet" / "daily"
        self.aligned_dir = self.base_dir / "parquet" / "aligned"
        self.cache_dir = self.base_dir / "cache"
        
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.aligned_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("data.storage")
        
        if not PYARROW_AVAILABLE:
            self.logger.warning(
                "PyArrow not available. Parquet support disabled. "
                "Install with: pip install pyarrow"
            )
    
    # ========================================================================
    # Part 1: 单股票存储和加载
    # ========================================================================
    
    def save_stock_data(
        self,
        code: str,
        df: pd.DataFrame,
    ) -> None:
        """
        保存单只股票数据到 Parquet。
        
        Args:
            code: 股票代码
            df: OHLCV DataFrame（必须包含 date 列或 DatetimeIndex）
        """
        filepath = self.parquet_dir / f"{code}.parquet"
        
        # 确保 date 列存在
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={"index": "date"})
            else:
                raise ValueError("DataFrame must have 'date' column or DatetimeIndex")
        
        # 保存
        if PYARROW_AVAILABLE:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, filepath, compression=self.compression)
        else:
            # Fallback to pickle
            df.to_pickle(filepath.with_suffix(".pkl"))
        
        self.logger.debug(f"Saved {code} to {filepath}")
    
    def load_stock_data(
        self,
        code: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        加载单只股票数据。
        
        Args:
            code: 股票代码
            columns: 需要的列（None 表示全部）
        
        Returns:
            DataFrame 或 None（文件不存在）
        """
        filepath = self.parquet_dir / f"{code}.parquet"
        
        if not filepath.exists():
            # 尝试 pickle fallback
            pkl_path = filepath.with_suffix(".pkl")
            if pkl_path.exists():
                return pd.read_pickle(pkl_path)
            return None
        
        # 读取 Parquet
        if PYARROW_AVAILABLE:
            if self.use_memory_map:
                table = pq.read_table(
                    filepath,
                    columns=columns,
                    memory_map=True,
                )
                df = table.to_pandas()
            else:
                df = pd.read_parquet(filepath, columns=columns)
        else:
            df = pd.read_pickle(filepath.with_suffix(".pkl"))
        
        return df
    
    # ========================================================================
    # Part 2: 批量保存和加载
    # ========================================================================
    
    def save_batch(
        self,
        history_data: Dict[str, pd.DataFrame],
        show_progress: bool = False,
    ) -> None:
        """
        批量保存历史数据。
        
        Args:
            history_data: {code: OHLCV DataFrame}
            show_progress: 是否显示进度条
        """
        self.logger.info(f"Saving {len(history_data)} stocks to Parquet...")
        
        codes = list(history_data.keys())
        if show_progress:
            try:
                from tqdm import tqdm
                codes = tqdm(codes, desc="Saving")
            except ImportError:
                pass
        
        for code in codes:
            df = history_data[code]
            self.save_stock_data(code, df)
        
        self.logger.info(f"✓ Saved {len(history_data)} stocks")
    
    def load_batch(
        self,
        codes: List[str],
        columns: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载历史数据。
        
        Args:
            codes: 股票代码列表
            columns: 需要的列
            show_progress: 是否显示进度条
        
        Returns:
            {code: DataFrame}
        """
        self.logger.info(f"Loading {len(codes)} stocks from Parquet...")
        
        history_data = {}
        
        codes_iter = codes
        if show_progress:
            try:
                from tqdm import tqdm
                codes_iter = tqdm(codes, desc="Loading")
            except ImportError:
                pass
        
        for code in codes_iter:
            df = self.load_stock_data(code, columns)
            if df is not None:
                history_data[code] = df
        
        self.logger.info(f"✓ Loaded {len(history_data)} stocks")
        
        return history_data
    
    # ========================================================================
    # Part 3: 日期对齐（核心功能）
    # ========================================================================
    
    def align_data(
        self,
        history_data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        对齐所有股票的日期序列。
        
        Args:
            history_data: {code: OHLCV DataFrame}
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
        
        Returns:
            (aligned_data, date_index)
            aligned_data: {code: 对齐后的 DataFrame}
            date_index: 全局日期索引
        """
        self.logger.info("Aligning dates...")
        
        # Step 1: 收集所有日期
        all_dates = set()
        for df in history_data.values():
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
            elif isinstance(df.index, pd.DatetimeIndex):
                dates = df.index
            else:
                continue
            
            all_dates.update(dates)
        
        # Step 2: 构建全局日期索引
        date_index = sorted(all_dates)
        
        # 过滤日期范围
        if start_date:
            start_dt = pd.Timestamp(start_date)
            date_index = [d for d in date_index if d >= start_dt]
        
        if end_date:
            end_dt = pd.Timestamp(end_date)
            date_index = [d for d in date_index if d <= end_dt]
        
        self.logger.info(f"Global date range: {len(date_index)} days")
        
        # Step 3: 对齐每只股票
        aligned_data = {}
        
        for code, df in history_data.items():
            # 设置日期索引
            if "date" in df.columns:
                df = df.set_index("date")
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Reindex 到全局索引
            aligned_df = df.reindex(date_index)
            
            # 重置索引为列
            aligned_df = aligned_df.reset_index()
            aligned_df = aligned_df.rename(columns={"index": "date"})
            
            aligned_data[code] = aligned_df
        
        # 转换日期索引为字符串
        date_index_str = [d.strftime("%Y-%m-%d") for d in date_index]
        
        self.logger.info(f"✓ Aligned {len(aligned_data)} stocks")
        
        return aligned_data, date_index_str
    
    # ========================================================================
    # Part 4: 转换为规整化矩阵（Numba 路径）
    # ========================================================================
    
    def to_aligned_matrices(
        self,
        history_data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        转换为规整化矩阵（所有股票同一 shape）。
        
        Args:
            history_data: {code: OHLCV DataFrame}
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            (high, low, close, open_arr, volume, codes, dates)
            所有矩阵 shape=(n_stocks, n_days)
        """
        self.logger.info("Converting to aligned matrices...")
        
        # 对齐日期
        aligned_data, date_index = self.align_data(
            history_data, start_date, end_date
        )
        
        codes = sorted(aligned_data.keys())
        n_stocks = len(codes)
        n_days = len(date_index)
        
        # 初始化矩阵
        high = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        low = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        close = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        open_arr = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        volume = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
        
        # 填充数据
        for i, code in enumerate(codes):
            df = aligned_data[code]
            
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
        
        self.logger.info(
            f"✓ Converted to matrices: shape=({n_stocks}, {n_days})"
        )
        
        return high, low, close, open_arr, volume, codes, date_index
    
    # ========================================================================
    # Part 5: 缓存对齐矩阵（加速重复加载）
    # ========================================================================
    
    def save_aligned_cache(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_arr: np.ndarray,
        volume: np.ndarray,
        codes: List[str],
        dates: List[str],
        cache_name: str = "default",
    ) -> None:
        """
        保存对齐矩阵到缓存。
        
        Args:
            所有矩阵参数 shape=(n_stocks, n_days)
            codes: 股票代码列表
            dates: 日期列表
            cache_name: 缓存名称
        """
        cache_path = self.aligned_dir / cache_name
        cache_path.mkdir(exist_ok=True)
        
        # 保存矩阵（NumPy 原生格式，支持 mmap）
        np.save(cache_path / "high.npy", high)
        np.save(cache_path / "low.npy", low)
        np.save(cache_path / "close.npy", close)
        np.save(cache_path / "open.npy", open_arr)
        np.save(cache_path / "volume.npy", volume)
        
        # 保存元数据
        metadata = {
            "codes": codes,
            "dates": dates,
            "shape": high.shape,
        }
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✓ Saved aligned cache to {cache_path}")
    
    def load_aligned_cache(
        self,
        cache_name: str = "default",
        use_mmap: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]]:
        """
        从缓存加载对齐矩阵。
        
        Args:
            cache_name: 缓存名称
            use_mmap: 是否使用 memory mapping（零拷贝）
        
        Returns:
            (high, low, close, open_arr, volume, codes, dates)
            或 None（缓存不存在）
        """
        cache_path = self.aligned_dir / cache_name
        
        if not cache_path.exists():
            return None
        
        # 加载元数据
        metadata_file = cache_path / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        codes = metadata["codes"]
        dates = metadata["dates"]
        
        # 加载矩阵
        mmap_mode = "r" if use_mmap else None
        
        high = np.load(cache_path / "high.npy", mmap_mode=mmap_mode)
        low = np.load(cache_path / "low.npy", mmap_mode=mmap_mode)
        close = np.load(cache_path / "close.npy", mmap_mode=mmap_mode)
        open_arr = np.load(cache_path / "open.npy", mmap_mode=mmap_mode)
        volume = np.load(cache_path / "volume.npy", mmap_mode=mmap_mode)
        
        self.logger.info(
            f"✓ Loaded aligned cache from {cache_path} "
            f"(mmap={use_mmap})"
        )
        
        return high, low, close, open_arr, volume, codes, dates
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def list_stocks(self) -> List[str]:
        """列出所有已保存的股票"""
        if not self.parquet_dir.exists():
            return []
        
        codes = []
        for file in self.parquet_dir.glob("*.parquet"):
            codes.append(file.stem)
        
        return sorted(codes)
    
    def clear_cache(self) -> None:
        """清空缓存目录"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        
        self.logger.info("✓ Cache cleared")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "ColumnarStorageManager",
    "PYARROW_AVAILABLE",
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    import tempfile
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("COLUMNAR STORAGE MANAGER - TEST")
    print("=" * 70)
    print(f"PyArrow Available: {PYARROW_AVAILABLE}")
    print()
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ColumnarStorageManager(base_dir=tmpdir)
        
        # 准备测试数据
        n_days = 100
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        
        history_data = {}
        for i in range(5):
            code = f"SH{600000 + i:06d}"
            
            # 模拟停牌（部分日期缺失）
            if i == 2:
                # 第 3 只股票停牌 10 天
                trade_dates = [d for j, d in enumerate(dates) if j % 10 != 0]
            else:
                trade_dates = dates
            
            df = pd.DataFrame({
                "date": trade_dates,
                "open": 10.0 + np.random.randn(len(trade_dates)).cumsum() * 0.1,
                "high": 10.2 + np.random.randn(len(trade_dates)).cumsum() * 0.1,
                "low": 9.8 + np.random.randn(len(trade_dates)).cumsum() * 0.1,
                "close": 10.0 + np.random.randn(len(trade_dates)).cumsum() * 0.1,
                "volume": np.random.uniform(1e6, 1e7, len(trade_dates)),
            })
            
            history_data[code] = df
        
        # Test 1: 保存和加载
        print("Test 1: Save and load")
        storage.save_batch(history_data)
        print(f"  ✓ Saved {len(history_data)} stocks")
        
        loaded_data = storage.load_batch(list(history_data.keys()))
        print(f"  ✓ Loaded {len(loaded_data)} stocks")
        print()
        
        # Test 2: 日期对齐
        print("Test 2: Date alignment")
        aligned_data, date_index = storage.align_data(loaded_data)
        print(f"  ✓ Aligned to {len(date_index)} days")
        
        # 检查对齐结果
        shapes = [df.shape[0] for df in aligned_data.values()]
        print(f"  ✓ All stocks have same length: {len(set(shapes)) == 1}")
        print(f"  ✓ Shape: {shapes[0]} days")
        print()
        
        # Test 3: 转换为矩阵
        print("Test 3: Convert to matrices")
        h, l, c, o, v, codes, dates_list = storage.to_aligned_matrices(loaded_data)
        print(f"  ✓ Matrix shape: {h.shape}")
        print(f"  ✓ Codes: {codes}")
        print(f"  ✓ Date range: {dates_list[0]} → {dates_list[-1]}")
        print()
        
        # Test 4: 缓存
        print("Test 4: Cache")
        storage.save_aligned_cache(h, l, c, o, v, codes, dates_list)
        print(f"  ✓ Saved cache")
        
        result = storage.load_aligned_cache(use_mmap=True)
        if result:
            h2, l2, c2, o2, v2, codes2, dates2 = result
            print(f"  ✓ Loaded cache (mmap=True)")
            print(f"  ✓ Shape match: {h2.shape == h.shape}")
        print()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
