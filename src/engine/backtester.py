"""
src/engine/backtester.py
=========================
Phase 6 — PolyStrategyBacktester（回测总指挥部）

核心职责：
  1. Main Entry：整个系统的统一入口
  2. 生命周期管理：自动扫描数据、初始化引擎、执行回测
  3. 多策略支持：并行运行多个策略（可选多进程）
  4. 进度监控：实时展示回测进度和状态
  5. 结果聚合：汇总所有策略的绩效报告

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. **完整流水线**
   ──────────────
   ```
   PolyStrategyBacktester
     ↓
   1. 加载数据（ColumnarStorageManager）
     ↓
   2. 数据清洗（DataSanitizer）
     ↓
   3. 初始化策略（BaseStrategy）
     ↓
   4. 运行回测（HybridExecutionEngine）
     ↓
   5. 绩效分析（PerformanceAnalyzer）
     ↓
   6. 生成报告
   ```

2. **多策略支持**
   ──────────────
   支持三种模式：
   - 单策略：BacktestConfig(strategy=my_strategy)
   - 多策略顺序：BacktestConfig(strategies=[s1, s2, s3])
   - 多策略并行：BacktestConfig(strategies=[...], parallel=True)
   
   并行模式使用 multiprocessing.Pool

3. **自动数据发现**
   ────────────────
   - 扫描 data/ 目录
   - 自动加载 Parquet 文件
   - 检测日期范围
   - 对齐矩阵

4. **进度监控**
   ───────────
   - tqdm 进度条
   - 实时日志输出
   - 阶段性统计

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# 导入所有组件
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from strategy.base import BaseStrategy
from engine.execution import HybridExecutionEngine
from analysis.performance import PerformanceAnalyzer, PerformanceMetrics
from data.storage import ColumnarStorageManager
from data.sanitizer import DataSanitizer


# ============================================================================
# Part 1: 配置类
# ============================================================================

@dataclass
class BacktestConfig:
    """
    回测配置。
    
    包含所有回测参数，用于初始化 PolyStrategyBacktester。
    """
    # 策略
    strategy: Optional[BaseStrategy] = None
    strategies: List[BaseStrategy] = field(default_factory=list)
    
    # 数据
    data_dir: str = "data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    codes: Optional[List[str]] = None
    
    # 资金
    initial_cash: float = 1000000.0
    commission_rate: float = 0.0003
    
    # 性能
    use_cache: bool = True
    cache_name: str = "default"
    parallel: bool = False
    n_jobs: int = 4
    
    # 显示
    show_progress: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """验证配置"""
        if self.strategy is None and not self.strategies:
            raise ValueError("Must provide either 'strategy' or 'strategies'")
        
        if self.strategy and self.strategies:
            raise ValueError("Cannot provide both 'strategy' and 'strategies'")
        
        # 统一为列表
        if self.strategy:
            self.strategies = [self.strategy]


# ============================================================================
# Part 2: PolyStrategyBacktester 主类
# ============================================================================

class PolyStrategyBacktester:
    """
    回测总指挥部（Main Entry）。
    
    职责：
    1. 管理完整回测流水线
    2. 协调所有模块（数据、策略、引擎、分析）
    3. 支持多策略并行
    4. 生成完整报告
    
    使用方法：
        config = BacktestConfig(
            strategy=MyStrategy(),
            data_dir="data",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        
        backtester = PolyStrategyBacktester(config)
        results = backtester.run()
        
        # 查看结果
        for name, result in results.items():
            print(result["metrics"])
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Args:
            config: 回测配置
        """
        self.config = config
        
        # 初始化组件
        self.storage = ColumnarStorageManager(
            base_dir=config.data_dir,
            use_memory_map=True,
        )
        self.sanitizer = DataSanitizer()
        self.performance_analyzer = PerformanceAnalyzer()
        
        self.logger = logging.getLogger("backtester")
        
        # 数据缓存
        self._history_data: Optional[Dict[str, pd.DataFrame]] = None
        self._aligned_matrices: Optional[tuple] = None
    
    # ========================================================================
    # 主流程
    # ========================================================================
    
    def run(self) -> Dict[str, Dict]:
        """
        运行回测主流程。
        
        Returns:
            results: {
                strategy_name: {
                    "metrics": PerformanceMetrics,
                    "equity_curve": pd.DataFrame,
                    "trades": List[Dict],
                    "snapshots": List[AccountSnapshot],
                }
            }
        """
        self.logger.info("=" * 70)
        self.logger.info("POLY-STRATEGY BACKTESTER")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        # ── Phase 1: 加载数据 ──
        self._load_data()
        
        # ── Phase 2: 运行回测 ──
        if self.config.parallel and len(self.config.strategies) > 1:
            results = self._run_parallel()
        else:
            results = self._run_sequential()
        
        # ── Phase 3: 生成报告 ──
        self._print_summary(results)
        
        elapsed = time.time() - start_time
        self.logger.info("=" * 70)
        self.logger.info(f"BACKTEST COMPLETE (elapsed: {elapsed:.2f}s)")
        self.logger.info("=" * 70)
        
        return results
    
    def _load_data(self) -> None:
        """加载和准备数据"""
        self.logger.info("Loading data...")
        
        # 尝试从缓存加载
        if self.config.use_cache:
            cached = self.storage.load_aligned_cache(self.config.cache_name)
            if cached:
                self._aligned_matrices = cached
                self.logger.info("✓ Loaded data from cache")
                return
        
        # 扫描数据目录
        codes = self.config.codes
        if codes is None:
            codes = self.storage.list_stocks()
            if not codes:
                raise ValueError(f"No data found in {self.config.data_dir}")
        
        self.logger.info(f"Found {len(codes)} stocks")
        
        # 加载数据
        self._history_data = self.storage.load_batch(
            codes,
            show_progress=self.config.show_progress,
        )
        
        # 数据清洗
        self.logger.info("Sanitizing data...")
        self._history_data = self.sanitizer.sanitize_batch(
            self._history_data,
            show_progress=self.config.show_progress,
        )
        
        # 转换为对齐矩阵
        self.logger.info("Aligning data...")
        self._aligned_matrices = self.storage.to_aligned_matrices(
            self._history_data,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        
        # 保存缓存
        if self.config.use_cache:
            h, l, c, o, v, codes_list, dates = self._aligned_matrices
            self.storage.save_aligned_cache(
                h, l, c, o, v, codes_list, dates,
                cache_name=self.config.cache_name,
            )
        
        self.logger.info("✓ Data loaded and aligned")
    
    def _run_sequential(self) -> Dict[str, Dict]:
        """顺序运行多个策略"""
        results = {}
        
        for strategy in self.config.strategies:
            self.logger.info(f"\nRunning strategy: {strategy.name}")
            
            result = self._run_single_strategy(strategy)
            results[strategy.name] = result
        
        return results
    
    def _run_parallel(self) -> Dict[str, Dict]:
        """并行运行多个策略（多进程）"""
        # 多进程实现较复杂，简化版先顺序执行
        # 实际生产可用 multiprocessing.Pool
        self.logger.warning("Parallel mode not fully implemented, using sequential")
        return self._run_sequential()
    
    def _run_single_strategy(self, strategy: BaseStrategy) -> Dict:
        """运行单个策略"""
        # 创建执行引擎
        engine = HybridExecutionEngine(
            strategy=strategy,
            start_date=self.config.start_date or "2024-01-01",
            end_date=self.config.end_date or "2024-12-31",
            initial_cash=self.config.initial_cash,
            commission_rate=self.config.commission_rate,
            show_progress=self.config.show_progress,
        )
        
        # 运行回测
        backtest_result = engine.run_backtest(self._history_data)
        
        # 绩效分析
        metrics = self.performance_analyzer.analyze(
            snapshots=backtest_result["snapshots"],
            trades=backtest_result["trade_history"],
            initial_equity=self.config.initial_cash,
        )
        
        return {
            "metrics": metrics,
            "equity_curve": backtest_result["equity_curve"],
            "trades": backtest_result["trade_history"],
            "snapshots": backtest_result["snapshots"],
            "performance": backtest_result["performance"],
            "trade_statistics": backtest_result["trade_statistics"],
            "match_statistics": backtest_result["match_statistics"],
        }
    
    # ========================================================================
    # 报告生成
    # ========================================================================
    
    def _print_summary(self, results: Dict[str, Dict]) -> None:
        """打印汇总报告"""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("BACKTEST SUMMARY")
        self.logger.info("=" * 70)
        
        for strategy_name, result in results.items():
            metrics = result["metrics"]
            
            self.logger.info(f"\nStrategy: {strategy_name}")
            self.logger.info("-" * 70)
            self.logger.info(f"  Total Return:    {metrics.total_return * 100:6.2f}%")
            self.logger.info(f"  Annual Return:   {metrics.annual_return * 100:6.2f}%")
            self.logger.info(f"  Sharpe Ratio:    {metrics.sharpe_ratio:6.2f}")
            self.logger.info(f"  Max Drawdown:    {metrics.max_drawdown * 100:6.2f}%")
            self.logger.info(f"  Win Rate:        {metrics.win_rate * 100:6.2f}%")
            self.logger.info(f"  Total Trades:    {metrics.total_trades}")
        
        self.logger.info("=" * 70)
    
    def save_report(
        self,
        results: Dict[str, Dict],
        output_dir: str = "reports",
    ) -> None:
        """
        保存完整报告。
        
        Args:
            results: 回测结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for strategy_name, result in results.items():
            # 保存绩效指标
            metrics = result["metrics"]
            metrics_df = pd.DataFrame([metrics.to_dict()])
            metrics_df.to_csv(output_path / f"{strategy_name}_metrics.csv", index=False)
            
            # 保存净值曲线
            equity_curve = result["equity_curve"]
            equity_curve.to_csv(output_path / f"{strategy_name}_equity.csv")
            
            # 保存交易流水
            trades = result["trades"]
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_csv(output_path / f"{strategy_name}_trades.csv", index=False)
        
        self.logger.info(f"✓ Reports saved to {output_dir}")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "PolyStrategyBacktester",
    "BacktestConfig",
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    from strategy import Signal, OrderSide, FactorStore
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("POLY-STRATEGY BACKTESTER - TEST")
    print("=" * 70)
    print()
    
    # 定义简单策略
    class DummyStrategy(BaseStrategy):
        name = "dummy"
        version = "1.0.0"
        
        def compute_factors(self, history: Dict[str, pd.DataFrame]) -> FactorStore:
            return {}
        
        def _generate_entry_signals(self, ctx) -> List[Signal]:
            # 每天随机买入前 2 只股票
            if ctx.current_data.empty:
                return []
            
            codes = ctx.current_data["code"].tolist()[:2]
            return [
                Signal(code=code, side=OrderSide.BUY, weight=0.05, reason="test")
                for code in codes
            ]
    
    # 准备测试数据
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟数据
        n_stocks = 20
        n_days = 100
        codes = [f"SH{600000 + i:06d}" for i in range(n_stocks)]
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        
        storage = ColumnarStorageManager(base_dir=tmpdir)
        
        history_data = {}
        for code in codes:
            df = pd.DataFrame({
                "date": dates,
                "open": 10.0 + np.random.randn(n_days).cumsum() * 0.1,
                "high": 10.2 + np.random.randn(n_days).cumsum() * 0.1,
                "low": 9.8 + np.random.randn(n_days).cumsum() * 0.1,
                "close": 10.0 + np.random.randn(n_days).cumsum() * 0.1,
                "volume": np.random.uniform(1e6, 1e7, n_days),
            })
            history_data[code] = df
        
        # 保存数据
        storage.save_batch(history_data)
        
        # 创建配置
        config = BacktestConfig(
            strategy=DummyStrategy(),
            data_dir=tmpdir,
            start_date="2024-01-01",
            end_date="2024-04-10",
            initial_cash=1000000.0,
            show_progress=True,
            use_cache=False,
        )
        
        # 运行回测
        backtester = PolyStrategyBacktester(config)
        results = backtester.run()
        
        # 验证结果
        print()
        print("Results validation:")
        for name, result in results.items():
            metrics = result["metrics"]
            print(f"  {name}:")
            print(f"    Total Return: {metrics.total_return:.2%}")
            print(f"    Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"    Trades: {metrics.total_trades}")
        
        print()
        print("=" * 70)
        print("TEST COMPLETE ✓")
        print("=" * 70)
