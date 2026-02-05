"""
src/data/collector.py
=====================
Phase 6 — TDX 并行数据采集器（v2.0.1 工业级重构版）

职责：
  1. 全市场 A 股日线数据采集（pytdx → parquet 全链路）
  2. 长连接池 + 负载均衡（15-30 线程高并发）
  3. 增量更新 + 前复权处理
  4. 契约式 ETL Pipeline: Download → Validate → Sanitize → Save

═══════════════════════════════════════════════════════════════════
架构对齐（v2.0.1 标准）
═══════════════════════════════════════════════════════════════════

1. **Path Hijacking（强契约）**
   ─────────────────────────
   storage.parquet_dir = base_dir/market_data/parquet/daily
   
   本模块 MUST NOT 重新定义路径，直接使用：
   ```python
   self.storage.parquet_dir  # 唯一路径源
   ```

2. **字段契约（严格对齐 types.py）**
   ──────────────────────────────
   DataSanitizer 要求字段：
     - open, high, low, close, volume (float64)
     - date (datetime64 index)
   
   TDX 原始字段：
     - vol → volume (必须立即 rename)
     - amount (保留，可选)
   
   rename 必须在 sanitize 之前完成，否则 DataSanitizer 报错。

3. **并发安全（CRITICAL FIX）**
   ───────────────────────────
   史前版本问题：
     - 日志竞态导致死锁（15 线程场景）
     - as_completed 未实时刷新进度
   
   v2.0.1 修复：
     - 使用 queue.Queue 异步日志（Logger 线程独立）
     - as_completed 每笔完成立即回调
     - 批量聚合日志（减少 I/O 竞争）

4. **Numba 安全性（对接 rsrs.py）**
   ─────────────────────────────────
   DataSanitizer.sanitize_dataframe() 保证：
     - NaN 修复（MAD 中值填充）
     - 价格异常值修复（forward fill）
     - volume < 100 修复（中值替换）
     - 数据类型强制 float64（Numba nopython 模式要求）

═══════════════════════════════════════════════════════════════════
性能目标
═══════════════════════════════════════════════════════════════════

- 节点测速: < 5s（30 并发）
- 下载速度: 15-20 stocks/s（15 线程）
- 全市场采集: 5500 stocks < 6 分钟

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, local
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# pytdx 依赖检查
try:
    from pytdx.hq import TdxHq_API
    PYTDX_AVAILABLE = True
except ImportError:
    PYTDX_AVAILABLE = False
    import warnings
    warnings.warn(
        "pytdx 未安装，数据采集功能不可用。安装: pip install pytdx",
        RuntimeWarning
    )

# 导入已修复的模块
from .storage import ColumnarStorageManager
from .sanitizer import DataSanitizer


# ============================================================================
# Part 1: 并发安全日志系统（CRITICAL FIX）
# ============================================================================

class AsyncLogHandler:
    """
    异步日志处理器（解决多线程日志死锁）。
    
    问题根源：
      - logging 模块在高并发下存在 GIL 竞争
      - 15 线程同时调用 logger.info() 导致 I/O 阻塞
    
    解决方案：
      - 主线程启动独立 Logger 线程
      - 工作线程通过 queue.Queue 异步发送日志
      - Logger 线程负责所有 I/O 操作
    """
    
    def __init__(self, logger: logging.Logger, max_queue_size: int = 10000):
        self.logger = logger
        self.log_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
    
    def start(self) -> None:
        """启动异步日志线程"""
        if self._is_running:
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._log_worker,
            name="AsyncLogger",
            daemon=True
        )
        self._thread.start()
        self._is_running = True
    
    def stop(self, timeout: float = 5.0) -> None:
        """停止异步日志线程"""
        if not self._is_running:
            return
        
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._is_running = False
    
    def _log_worker(self) -> None:
        """日志工作线程（独立线程，无 GIL 竞争）"""
        while not self._stop_event.is_set():
            try:
                # 0.1s 超时，避免退出时卡住
                level, msg = self.log_queue.get(timeout=0.1)
                
                if level == logging.DEBUG:
                    self.logger.debug(msg)
                elif level == logging.INFO:
                    self.logger.info(msg)
                elif level == logging.WARNING:
                    self.logger.warning(msg)
                elif level == logging.ERROR:
                    self.logger.error(msg)
                
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # 降级到同步日志（避免日志系统本身崩溃）
                print(f"[AsyncLogger ERROR] {e}", file=sys.stderr)
    
    def log(self, level: int, msg: str) -> None:
        """异步发送日志（非阻塞）"""
        try:
            self.log_queue.put_nowait((level, msg))
        except queue.Full:
            # 队列满时降级到同步日志
            self.logger.log(level, msg)
    
    def debug(self, msg: str) -> None:
        self.log(logging.DEBUG, msg)
    
    def info(self, msg: str) -> None:
        self.log(logging.INFO, msg)
    
    def warning(self, msg: str) -> None:
        self.log(logging.WARNING, msg)
    
    def error(self, msg: str) -> None:
        self.log(logging.ERROR, msg)


# ============================================================================
# Part 2: 数据结构定义
# ============================================================================

@dataclass
class TdxNode:
    """TDX 服务器节点（immutable for thread safety）"""
    name: str
    host: str
    port: int
    latency: float = float('inf')
    is_available: bool = False
    fail_count: int = 0


@dataclass
class DownloadResult:
    """下载结果（契约输出）"""
    code: str
    success: bool
    records: int = 0
    message: str = ""
    elapsed_time: float = 0.0
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        if self.success and self.records > 0:
            return f"{status} {self.code} | {self.records} records | {self.elapsed_time:.2f}s"
        else:
            return f"{status} {self.code} | {self.message}"


# ============================================================================
# Part 3: TDX 节点管理器
# ============================================================================

class TdxNodeManager:
    """
    TDX 节点管理器（负载均衡 + 故障转移）。
    
    关键算法：
      - 轮询调度: worker_id % len(available_nodes)
      - 故障转移: fail_count >= 5 自动禁用
      - 自动恢复: 成功时递减 fail_count
    """
    
    # 精选高可用节点（经生产验证）
    DEFAULT_NODES: List[Tuple[str, str, int]] = [
        # 一线券商（最稳定）
        ("招商证券深圳", "119.147.212.81", 7709),
        ("华泰证券上海", "180.153.39.51", 7709),
        ("国信证券深圳", "120.79.60.82", 7709),
        ("中信证券上海", "101.227.73.20", 7709),
        ("银河证券北京", "106.120.74.86", 7709),
        ("广发证券深圳", "14.17.75.71", 7709),
        ("国泰君安上海", "180.153.18.170", 7709),
        ("海通证券杭州", "115.238.56.198", 7709),
        # 通达信主站
        ("通达信主站1", "110.41.147.114", 7709),
        ("通达信主站2", "221.194.181.176", 7709),
        ("通达信主站3", "59.175.238.38", 7709),
        ("通达信高带A", "112.74.214.43", 7721),
        ("通达信高带B", "120.24.149.28", 7721),
        # 备用节点
        ("东方财富上海", "183.136.120.48", 7709),
        ("平安证券深圳", "113.105.142.136", 7709),
    ]
    
    def __init__(self, timeout: float = 3.0, max_fail_count: int = 5):
        self.timeout = timeout
        self.max_fail_count = max_fail_count
        self.nodes: List[TdxNode] = [
            TdxNode(name=name, host=host, port=port)
            for name, host, port in self.DEFAULT_NODES
        ]
        self._lock = Lock()
    
    def ping_node(self, node: TdxNode) -> TdxNode:
        """测试单个节点延迟"""
        if not PYTDX_AVAILABLE:
            return node
        
        api = TdxHq_API()
        try:
            start = time.perf_counter()
            if api.connect(node.host, node.port, time_out=self.timeout):
                count = api.get_security_count(0)
                if count and count > 0:
                    node.latency = (time.perf_counter() - start) * 1000
                    node.is_available = True
                    node.fail_count = 0
        except Exception:
            node.is_available = False
        finally:
            try:
                api.disconnect()
            except Exception as e:
                pass
        
        return node
    
    def test_all_nodes(
        self,
        max_workers: int = 30,
        async_logger: Optional[AsyncLogHandler] = None
    ) -> List[TdxNode]:
        """并行测试所有节点（30 并发 < 5s）"""
        logger_obj = async_logger if async_logger else logging.getLogger(__name__)
        
        logger_obj.info("🔍 开始测试 TDX 节点...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(len(self.nodes), max_workers)) as executor:
            futures = {executor.submit(self.ping_node, node): node for node in self.nodes}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass
        
        available = sorted(
            [n for n in self.nodes if n.is_available],
            key=lambda x: x.latency
        )
        
        elapsed = time.time() - start_time
        logger_obj.info(
            f"✅ 节点测试完成: {len(available)}/{len(self.nodes)} 可用 | "
            f"耗时 {elapsed:.2f}s"
        )
        
        if available:
            top5 = ", ".join(f"{n.name}({n.latency:.0f}ms)" for n in available[:5])
            logger_obj.info(f"🚀 最快节点: {top5}")
        
        return available
    
    def get_node_by_index(self, index: int) -> Optional[TdxNode]:
        """轮询调度（负载均衡核心）"""
        with self._lock:
            available = [
                n for n in self.nodes
                if n.is_available and n.fail_count < self.max_fail_count
            ]
            if not available:
                return None
            return available[index % len(available)]
    
    def get_available_count(self) -> int:
        """获取可用节点数"""
        with self._lock:
            return len([
                n for n in self.nodes
                if n.is_available and n.fail_count < self.max_fail_count
            ])
    
    def report_failure(self, node: TdxNode) -> None:
        """报告节点失败（线程安全）"""
        with self._lock:
            node.fail_count += 1
            if node.fail_count >= self.max_fail_count:
                node.is_available = False
    
    def report_success(self, node: TdxNode) -> None:
        """报告节点成功（自动恢复）"""
        with self._lock:
            if node.fail_count > 0:
                node.fail_count = max(0, node.fail_count - 1)


# ============================================================================
# Part 4: 前复权处理器
# ============================================================================

class ForwardAdjustmentProcessor:
    """前复权处理器（静态工具类）"""
    
    @staticmethod
    def apply_forward_adjust(
        df: pd.DataFrame,
        xdxr_data: List[Dict]
    ) -> pd.DataFrame:
        """
        应用前复权。
        
        算法：
          - 从最新除权日向历史回溯
          - factor = 1 + 送转股比例 + 配股比例
          - adjusted_price = (原价 - 分红) / factor
        """
        if not xdxr_data or df.empty:
            return df
        
        try:
            xdxr_df = pd.DataFrame(xdxr_data)
            xdxr_df = xdxr_df[xdxr_df['category'] == 1].copy()
            
            if xdxr_df.empty:
                return df
            
            # 解析除权日期
            xdxr_df['date'] = pd.to_datetime(
                xdxr_df['year'].astype(str) + '-' +
                xdxr_df['month'].astype(str).str.zfill(2) + '-' +
                xdxr_df['day'].astype(str).str.zfill(2)
            )
            xdxr_df = xdxr_df.sort_values('date')
            
            result = df.copy()
            price_cols = ['open', 'high', 'low', 'close']
            
            # 从最新除权日向历史回溯
            for _, row in xdxr_df.iloc[::-1].iterrows():
                ex_date = row['date']
                songzhuangu = float(row.get('songzhuangu', 0) or 0) / 10
                peigu = float(row.get('peigu', 0) or 0) / 10
                fenhong = float(row.get('fenhong', 0) or 0) / 10
                factor = 1 + songzhuangu + peigu
                
                if factor > 0:
                    mask = result.index < ex_date
                    for col in price_cols:
                        if col in result.columns:
                            result.loc[mask, col] = (
                                result.loc[mask, col] - fenhong
                            ) / factor
            
            # 价格下限保护
            for col in price_cols:
                if col in result.columns:
                    result[col] = result[col].clip(lower=0.01)
            
            return result
        
        except Exception as e:
            logging.getLogger(__name__).debug(f"前复权处理异常: {e}")
            return df


# ============================================================================
# Part 5: 线程本地 API 管理器（长连接池）
# ============================================================================

class ThreadLocalAPI:
    """线程本地 API 管理器（长连接复用）"""
    
    def __init__(self):
        self._local = local()
    
    def get_api(self) -> Optional[TdxHq_API]:
        return getattr(self._local, 'api', None)
    
    def get_node(self) -> Optional[TdxNode]:
        return getattr(self._local, 'node', None)
    
    def set_connection(self, api: TdxHq_API, node: TdxNode) -> None:
        self._local.api = api
        self._local.node = node
    
    def clear(self) -> None:
        if hasattr(self._local, 'api'):
            try:
                self._local.api.disconnect()
            except Exception as e:
                pass
            delattr(self._local, 'api')
        if hasattr(self._local, 'node'):
            delattr(self._local, 'node')


# ============================================================================
# Part 6: TDX 并行下载器（主类）
# ============================================================================

class TdxParallelDownloader:
    """
    TDX 并行数据下载器 v2.0.1（工业级重构版）。
    
    架构对齐：
      - Path: 使用 storage.parquet_dir（强契约）
      - ETL: Download → Validate → Sanitize → Save
      - 并发: AsyncLogger + as_completed 实时进度
      - 安全: DataSanitizer 保证 Numba 兼容性
    
    性能目标：
      - 15-20 stocks/s (15 线程)
      - 5500 stocks < 6 分钟
    """
    
    # 有效股票代码前缀（契约）
    VALID_PREFIXES_SZ = ('00', '30', '8', '43')
    VALID_PREFIXES_SH = ('60', '688')
    
    def __init__(
        self,
        storage_manager: ColumnarStorageManager,
        data_sanitizer: DataSanitizer,
        max_workers: int = 15,
        timeout: float = 5.0,
        enable_adjust: bool = True,
        enable_async_log: bool = True
    ):
        """
        初始化下载器。
        
        Args:
            storage_manager: 存储管理器（路径契约来源）
            data_sanitizer: 数据清洗器（Numba 安全性保证）
            max_workers: 最大工作线程数（建议 15-30）
            timeout: 连接超时（秒）
            enable_adjust: 是否启用前复权
            enable_async_log: 是否启用异步日志（高并发必须）
        """
        if not PYTDX_AVAILABLE:
            raise RuntimeError("pytdx 未安装，无法使用数据采集功能")
        
        self.storage = storage_manager
        self.sanitizer = data_sanitizer
        self.max_workers = max_workers
        self.timeout = timeout
        self.enable_adjust = enable_adjust
        
        # ================================================================
        # 【契约】Path Hijacking - 直接使用 storage.parquet_dir
        # ================================================================
        self.parquet_dir = self.storage.parquet_dir
        
        # 初始化组件
        self.node_manager = TdxNodeManager(timeout=timeout)
        self.adjust_processor = ForwardAdjustmentProcessor()
        self.thread_local_api = ThreadLocalAPI()
        
        # 【CRITICAL FIX】异步日志系统
        self._base_logger = logging.getLogger(__name__)
        if enable_async_log:
            self.logger = AsyncLogHandler(self._base_logger)
            self.logger.start()
        else:
            self.logger = self._base_logger  # type: ignore
        
        # 统计信息（线程安全）
        self._stats: Dict[str, Any] = {}
        self._stats_lock = Lock()
        
        self.logger.info(f"📂 数据存储路径: {self.parquet_dir}")
    
    def __del__(self):
        """析构时停止异步日志"""
        if isinstance(self.logger, AsyncLogHandler):
            self.logger.stop()
    
    def test_nodes(self) -> List[TdxNode]:
        """测试所有节点"""
        return self.node_manager.test_all_nodes(
            max_workers=30,
            async_logger=self.logger if isinstance(self.logger, AsyncLogHandler) else None
        )
    
    def get_all_stock_codes(self) -> List[Tuple[int, str]]:
        """
        获取全市场 A 股代码列表。
        
        契约：
          - 深圳(0): 00/30/8/43
          - 上海(1): 60/688
        """
        node = self.node_manager.get_node_by_index(0)
        if node is None:
            self.test_nodes()
            node = self.node_manager.get_node_by_index(0)
        if node is None:
            raise RuntimeError("没有可用的 TDX 节点")
        
        self.logger.info("📋 正在获取全市场股票列表...")
        
        all_stocks = []
        api = TdxHq_API()
        
        try:
            if not api.connect(node.host, node.port, time_out=self.timeout):
                raise ConnectionError(f"无法连接到节点 {node.name}")
            
            for market in [0, 1]:
                count = api.get_security_count(market)
                if not count or count <= 0:
                    continue
                
                for start in range(0, count, 1000):
                    stocks = api.get_security_list(market, start)
                    if not stocks:
                        continue
                    
                    for stock in stocks:
                        code = stock['code']
                        
                        if market == 0:
                            if any(code.startswith(p) for p in self.VALID_PREFIXES_SZ):
                                all_stocks.append((market, code))
                        else:
                            if any(code.startswith(p) for p in self.VALID_PREFIXES_SH):
                                all_stocks.append((market, code))
        
        finally:
            try:
                api.disconnect()
            except Exception as e:
                pass
        
        self.logger.info(
            f"✅ 获取到 {len(all_stocks)} 只 A 股 | "
            f"深圳: {sum(1 for m, _ in all_stocks if m == 0)} | "
            f"上海: {sum(1 for m, _ in all_stocks if m == 1)}"
        )
        
        return all_stocks
    
    def _ensure_connection(
        self,
        worker_id: int
    ) -> Tuple[Optional[TdxHq_API], Optional[TdxNode]]:
        """确保当前线程有可用连接（长连接复用）"""
        api = self.thread_local_api.get_api()
        node = self.thread_local_api.get_node()
        
        # 验证现有连接
        if api and node:
            try:
                if api.get_security_count(0):
                    return api, node
            except Exception as e:
                pass
            
            self.thread_local_api.clear()
        
        # 轮询分配节点
        node = self.node_manager.get_node_by_index(worker_id)
        if not node:
            return None, None
        
        # 建立新连接
        api = TdxHq_API()
        try:
            if api.connect(node.host, node.port, time_out=self.timeout):
                self.thread_local_api.set_connection(api, node)
                return api, node
        except Exception:
            self.node_manager.report_failure(node)
        
        return None, None
    
    def _download_stock_data(
        self,
        api: TdxHq_API,
        market: int,
        code: str,
        start_date: Optional[datetime] = None
    ) -> Tuple[Optional[pd.DataFrame], List[Dict]]:
        """
        下载单只股票数据。
        
        契约：
          - 使用 count=800 批量获取
          - 立即 rename vol → volume
          - 返回 DataFrame + 除权数据
        """
        all_data = []
        xdxr_data = []
        
        # 批量获取（800 最优）
        start = 0
        while True:
            data = api.get_security_bars(
                category=9,
                market=market,
                code=code,
                start=start,
                count=800
            )
            
            if not data:
                break
            
            all_data.extend(data)
            start += 800
            
            if len(data) < 800:
                break
        
        # 获取除权除息
        if self.enable_adjust:
            xdxr_data = api.get_xdxr_info(market, code) or []
        
        if not all_data:
            return None, xdxr_data
        
        # 转换为 DataFrame
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['datetime'].str[:10])
        
        # ================================================================
        # 【契约】CRITICAL: 立即 rename vol → volume
        # ================================================================
        if 'vol' in df.columns:
            df = df.rename(columns={'vol': 'volume'})
        
        # 选择需要的列（契约字段）
        columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = df[[c for c in columns if c in df.columns]]
        df = df.set_index('date').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        # 应用前复权
        if self.enable_adjust and xdxr_data:
            df = self.adjust_processor.apply_forward_adjust(df, xdxr_data)
        
        # 增量过滤
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        
        return df, xdxr_data
    
    def _download_worker(
        self,
        tasks: List[Tuple[int, str]],
        worker_id: int
    ) -> List[DownloadResult]:
        """
        工作线程（处理一批股票）。
        
        ETL Pipeline:
          1. Download（TDX API）
          2. Validate（字段检查）
          3. Sanitize（DataSanitizer）
          4. Save（storage.save_stock_data）
        """
        results = []
        
        try:
            for market, code in tasks:
                start_time = time.time()
                
                try:
                    # 确保连接
                    api, node = self._ensure_connection(worker_id)
                    if not api or not node:
                        results.append(DownloadResult(
                            code=code,
                            success=False,
                            message="无可用连接",
                            elapsed_time=time.time() - start_time
                        ))
                        continue
                    
                    # 检查增量更新
                    last_date = self._get_last_local_date(code)
                    start_date = None
                    is_incremental = False
                    
                    if last_date:
                        start_date = last_date + timedelta(days=1)
                        if start_date.date() >= datetime.now().date():
                            results.append(DownloadResult(
                                code=code,
                                success=True,
                                records=0,
                                message="已是最新",
                                elapsed_time=time.time() - start_time
                            ))
                            continue
                        is_incremental = True
                    
                    # ========================================================
                    # ETL Pipeline
                    # ========================================================
                    
                    # Step 1: Download
                    df, _ = self._download_stock_data(api, market, code, start_date)
                    
                    if df is None or df.empty:
                        self.node_manager.report_success(node)
                        results.append(DownloadResult(
                            code=code,
                            success=True,
                            records=0,
                            message="无新数据" if is_incremental else "无数据",
                            elapsed_time=time.time() - start_time
                        ))
                        continue
                    
                    # Step 2: Validate（字段契约检查）
                    required = ['open', 'high', 'low', 'close', 'volume']
                    missing = [c for c in required if c not in df.columns]
                    if missing:
                        raise ValueError(f"缺少必需字段: {missing}")
                    
                    # Step 3: Sanitize（Numba 安全性保证）
                    df = self.sanitizer.sanitize_dataframe(df)
                    
                    # Step 4: Save
                    self.storage.save_stock_data(code, df)
                    
                    self.node_manager.report_success(node)
                    results.append(DownloadResult(
                        code=code,
                        success=True,
                        records=len(df),
                        message="成功",
                        elapsed_time=time.time() - start_time
                    ))
                
                except Exception as e:
                    node = self.thread_local_api.get_node()
                    if node:
                        self.node_manager.report_failure(node)
                    
                    results.append(DownloadResult(
                        code=code,
                        success=False,
                        message=str(e)[:50],
                        elapsed_time=time.time() - start_time
                    ))
                    
                    # 清理断开的连接
                    self.thread_local_api.clear()
        
        finally:
            self.thread_local_api.clear()
        
        return results
    
    def _get_last_local_date(self, code: str) -> Optional[datetime]:
        """获取本地存储的最后日期"""
        try:
            file_path = self.parquet_dir / f"{code}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                if not df.empty:
                    if 'date' in df.columns:
                        return pd.to_datetime(df['date'].max())
                    elif df.index.name == 'date':
                        return pd.to_datetime(df.index.max())
        except Exception:
            pass
        return None
    
    def download_all_stocks(
        self,
        stock_list: Optional[List[Tuple[int, str]]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        batch_log_size: int = 50
    ) -> Dict[str, Any]:
        """
        并行下载全市场股票。
        
        Args:
            stock_list: 股票列表（None 则自动获取）
            progress_callback: 进度回调 (current, total, code)
            batch_log_size: 批量日志间隔
        
        Returns:
            统计结果字典
        """
        # 确保有可用节点
        if self.node_manager.get_available_count() == 0:
            self.test_nodes()
        
        if self.node_manager.get_available_count() == 0:
            raise RuntimeError("没有可用的 TDX 节点")
        
        # 获取股票列表
        if stock_list is None:
            stock_list = self.get_all_stock_codes()
        
        total = len(stock_list)
        available_count = self.node_manager.get_available_count()
        effective_workers = min(self.max_workers, available_count * 3, total)
        
        # 初始化统计
        self._stats = {
            'total': total,
            'success': 0,
            'skip': 0,
            'fail': 0,
            'start_time': datetime.now(),
            'end_time': None,
            'total_records': 0
        }
        
        self.logger.info("=" * 70)
        self.logger.info(f"🚀 开始下载 {total} 只股票")
        self.logger.info(f"   工作线程: {effective_workers} | 可用节点: {available_count}")
        self.logger.info(f"   存储路径: {self.parquet_dir}")
        self.logger.info("=" * 70)
        
        # 任务分配（负载均衡）
        tasks_per_worker = [[] for _ in range(effective_workers)]
        for idx, stock in enumerate(stock_list):
            tasks_per_worker[idx % effective_workers].append(stock)
        
        completed = 0
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(self._download_worker, tasks, wid): wid
                for wid, tasks in enumerate(tasks_per_worker) if tasks
            }
            
            # ============================================================
            # 【CRITICAL FIX】as_completed 实时进度刷新
            # ============================================================
            for future in as_completed(futures):
                worker_id = futures[future]
                
                try:
                    results = future.result()
                    
                    for result in results:
                        completed += 1
                        
                        # 更新统计（线程安全）
                        with self._stats_lock:
                            if result.success:
                                if result.records > 0:
                                    self._stats['success'] += 1
                                    self._stats['total_records'] += result.records
                                else:
                                    self._stats['skip'] += 1
                            else:
                                self._stats['fail'] += 1
                        
                        # 回调
                        if progress_callback:
                            progress_callback(completed, total, result.code)
                        
                        # 批量日志（减少 I/O）
                        if completed % batch_log_size == 0 or completed == total:
                            elapsed = (datetime.now() - self._stats['start_time']).total_seconds()
                            speed = completed / elapsed if elapsed > 0 else 0
                            eta = (total - completed) / speed if speed > 0 else 0

                            self.logger.info(
                                f"📊 进度: {completed}/{total} ({completed/total*100:.1f}%) | "
                                f"成功: {self._stats['success']} | "
                                f"跳过: {self._stats['skip']} | "
                                f"失败: {self._stats['fail']} | "
                                f"速度: {speed:.1f}/s | "
                                f"剩余: {eta:.0f}s"
                            )
                            sys.stdout.flush()
                
                except Exception as e:
                    self.logger.error(f"Worker-{worker_id} 异常: {e}")
        
        self._stats['end_time'] = datetime.now()
        elapsed = (self._stats['end_time'] - self._stats['start_time']).total_seconds()
        
        self.logger.info("=" * 70)
        self.logger.info("✅ 下载完成!")
        self.logger.info(
            f"   总计: {total} | 成功: {self._stats['success']} | "
            f"跳过: {self._stats['skip']} | 失败: {self._stats['fail']}"
        )
        self.logger.info(f"   总记录数: {self._stats['total_records']:,}")
        self.logger.info(f"   耗时: {elapsed:.1f}s | 平均速度: {total/elapsed:.1f} 只/秒")
        self.logger.info("=" * 70)
        
        return self._stats.copy()
    
    def download_single(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """下载单只股票（便捷方法）"""
        if self.node_manager.get_available_count() == 0:
            self.test_nodes()
        
        # 判断市场
        if any(code.startswith(p) for p in self.VALID_PREFIXES_SZ):
            market = 0
        elif any(code.startswith(p) for p in self.VALID_PREFIXES_SH):
            market = 1
        else:
            raise ValueError(f"无法识别的股票代码: {code}")
        
        node = self.node_manager.get_node_by_index(0)
        if not node:
            raise RuntimeError("没有可用的 TDX 节点")
        
        api = TdxHq_API()
        try:
            if not api.connect(node.host, node.port, time_out=self.timeout):
                raise ConnectionError(f"无法连接到节点 {node.name}")
            
            start = pd.to_datetime(start_date) if start_date else None
            
            df, _ = self._download_stock_data(api, market, code, start)
            
            if df is not None and not df.empty:
                # Sanitize
                df = self.sanitizer.sanitize_dataframe(df)
                
                # 日期过滤
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]
            
            return df
        
        finally:
            try:
                api.disconnect()
            except Exception as e:
                pass


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'TdxParallelDownloader',
    'TdxNodeManager',
    'AsyncLogHandler',
    'DownloadResult',
    'PYTDX_AVAILABLE',
]


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("\n" + "=" * 70)
    print("TDX 并行数据采集器 v2.0.1（工业级重构版）")
    print("=" * 70)
    
    # 初始化组件
    storage = ColumnarStorageManager(base_dir="./data")
    sanitizer = DataSanitizer()
    
    # 创建下载器
    downloader = TdxParallelDownloader(
        storage_manager=storage,
        data_sanitizer=sanitizer,
        max_workers=15,
        timeout=5.0,
        enable_adjust=True,
        enable_async_log=True  # 高并发必须
    )
    
    # 测试节点
    print("\n[1/3] 测试 TDX 节点...")
    available = downloader.test_nodes()
    print(f"✓ 可用节点: {len(available)}")
    
    # 单股票测试
    print("\n[2/3] 单股票下载测试...")
    df = downloader.download_single("000001", start_date="2024-01-01")
    if df is not None:
        print(f"✓ 000001 下载成功: {len(df)} 条记录")
        print(df.tail(3))
    
    # 批量测试
    print("\n[3/3] 批量下载测试...")
    test_codes = ['000001', '000002', '600000', '600036']
    stock_list = []
    for code in test_codes:
        if code.startswith(('00', '30')):
            stock_list.append((0, code))
        else:
            stock_list.append((1, code))
    
    stats = downloader.download_all_stocks(stock_list)
    print(f"✓ 批量下载完成: 成功 {stats['success']}, 失败 {stats['fail']}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
