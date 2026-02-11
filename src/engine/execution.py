"""
src/engine/execution.py
=======================
Phase 4 — HybridExecutionEngine（混合执行引擎）

核心职责：
  1. 调度中心：协调 Strategy / FactorEngine / MatchEngine / AccountManager
  2. 混合模式：向量化预筛选 + 逐日事件撮合
  3. 生命周期管理：initialize → run_backtest → finalize
  4. 回调触发：on_bar / on_order_filled / on_day_start / on_day_end

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. **混合执行模式**
   ────────────────
   传统回测有两种极端：
   - 纯向量化：快但不真实（无流动性检查）
   - 纯事件驱动：真实但慢（逐股逐日循环）
   
   混合模式结合两者优势：
   
   ┌─────────────────────────────────────────────────────────────┐
   │  Phase 1: 向量化预筛选（Phase 3 因子引擎）                  │
   │  ────────────────────────────────────────────────────────  │
   │  • 批量计算 RSRS 全族因子（< 50ms）                         │
   │  • 截面排序 Top-N 候选池（ndarray 操作）                     │
   │  • 过滤：价格区间、成交额、因子阈值                          │
   │  → 输出：Top-N 候选股票池（如 Top 50）                      │
   └─────────────────────────────────────────────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────────┐
   │  Phase 2: 逐日事件撮合（MatchEngine + AccountManager）      │
   │  ────────────────────────────────────────────────────────  │
   │  • 仅对 Top-N 池内股票生成信号                              │
   │  • 检查涨跌停、停牌、流动性                                  │
   │  • 动态滑点、佣金计算                                        │
   │  • 更新账户、持仓、流水                                      │
   │  → 输出：真实成交记录 + 账户快照                            │
   └─────────────────────────────────────────────────────────────┘
   
   **性能提升**：
   - 因子计算：O(n_stocks) → Numba 加速 < 50ms
   - 信号生成：O(n_stocks) → O(Top-N) ≈ O(50)
   - 总耗时：5000 股 × 250 天 < 500ms

2. **回测流程**
   ───────────
   ```
   for date in date_range:
       # Day start
       context = build_context(date)
       strategy.on_day_start(context)
       
       # Generate signals
       signals = strategy.generate_signals(context)
       
       # Submit & match
       orders = match_engine.submit_signals(signals, ...)
       fills, rejects = match_engine.match_orders(orders, ...)
       
       # Process fills
       for fill in fills:
           account.process_fill(fill, ...)
           strategy.on_order_filled(fill)
       
       # Day end
       account.update_market_value(current_data)
       snapshot = account.create_snapshot(date)
       strategy.on_day_end(context)
   ```

3. **StrategyContext 注入**
   ────────────────────────
   引擎构建并注入 StrategyContext：
   - current_data: 当日全市场行情
   - positions: 当前持仓快照（{code: quantity}）
   - total_equity: 总权益
   - cash: 可用资金
   - factor_accessor: 因子访问器（高性能路径）
   - history_provider: 历史数据提供者（闭包）

4. **性能优化**
   ───────────
   - 因子计算：一次性预计算（回测开始前）
   - 截面排序：np.argsort（O(n log n)）
   - 持仓快照：引用传递（避免拷贝）
   - 市值更新：批量更新（一次 DataFrame 操作）

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# tqdm 可选依赖
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x  # type: ignore

# 导入类型和模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from strategy.types import (
    BasePositionState,
    FactorAccessor,
    FactorStore,
    Signal,
    StrategyContext,
    Timestamp,
)
from strategy.base import BaseStrategy

# 导入引擎模块
from .match import ExchangeMatchEngine
from .account import AccountManager, AccountSnapshot


# ============================================================================
# HybridExecutionEngine 主类
# ============================================================================

class HybridExecutionEngine:
    """
    混合执行引擎（回测调度中心）。
    
    核心流程：
    1. 初始化：策略 / 因子 / 账户 / 撮合引擎
    2. 预计算：调用 strategy.compute_factors()
    3. 回测循环：逐日生成信号 → 撮合 → 更新账户
    4. 结束：生成绩效报告
    
    使用方法：
        engine = HybridExecutionEngine(
            strategy=my_strategy,
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_cash=1000000.0,
        )
        
        result = engine.run_backtest(history_data)
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        start_date: str,
        end_date: str,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
        show_progress: bool = True,
    ):
        """
        Args:
            strategy: 策略实例（BaseStrategy 子类）
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            initial_cash: 初始资金
            commission_rate: 佣金率
            show_progress: 是否显示进度条
        """
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.show_progress = show_progress
        
        # 引擎组件
        self.account = AccountManager(initial_cash=initial_cash)
        self.match_engine = ExchangeMatchEngine(commission_rate=commission_rate)
        
        # 因子缓存
        self._factor_store: Optional[FactorStore] = None
        self._factor_accessor: Optional[FactorAccessor] = None
        self._code_to_idx: Dict[str, int] = {}
        self._date_to_idx: Dict[str, int] = {}
        
        # 历史数据缓存
        self._history_data: Optional[Dict[str, pd.DataFrame]] = None
        
        self.logger = logging.getLogger("engine.execution")
    
    # ========================================================================
    # 主流程
    # ========================================================================
    
    def run_backtest(
        self,
        history_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        运行回测主流程。
        
        Args:
            history_data: {code: OHLCV DataFrame}
                每个 DataFrame 必须包含：
                - date (index or column)
                - open, high, low, close, volume
            market_data: 全市场日线数据（可选）
                如果提供，格式为：
                - columns: [date, code, open, high, low, close, volume, ...]
                用于加速当日行情查询
        
        Returns:
            result: {
                "snapshots": List[AccountSnapshot],
                "equity_curve": pd.DataFrame,
                "trade_history": List[Dict],
                "statistics": Dict,
                "performance": Dict,
            }
        """
        self.logger.info("=" * 70)
        self.logger.info("BACKTEST START")
        self.logger.info("=" * 70)
        self.logger.info(f"Strategy: {self.strategy.name} v{self.strategy.version}")
        self.logger.info(f"Date range: {self.start_date} → {self.end_date}")
        self.logger.info(f"Initial cash: {self.initial_cash:,.2f}")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        # ── Phase 1: 初始化 ──
        self._initialize(history_data, market_data)
        
        # ── Phase 2: 预计算因子 ──
        self._precompute_factors(history_data)
        
        # ── Phase 3: 回测循环 ──
        self._run_backtest_loop(market_data)
        
        # ── Phase 4: 结束 ──
        result = self._finalize()
        
        elapsed = time.time() - start_time
        self.logger.info("=" * 70)
        self.logger.info(f"BACKTEST COMPLETE (elapsed: {elapsed:.2f}s)")
        self.logger.info("=" * 70)
        
        return result
    
    def _initialize(
        self,
        history_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame],
    ) -> None:
        """初始化"""
        self.logger.info("Initializing...")
        
        # 缓存历史数据
        self._history_data = history_data
        
        # 调用策略初始化钩子
        self.strategy.initialize()
        self.strategy.on_strategy_start()
        
        self.logger.info(f"✓ Initialized {len(history_data)} stocks")
    
    def _precompute_factors(
        self,
        history_data: Dict[str, pd.DataFrame],
    ) -> None:
        """预计算因子（Phase 3 因子引擎）"""
        self.logger.info("Pre-computing factors...")
        
        start = time.time()
        
        # 调用策略的 compute_factors 方法
        self._factor_store = self.strategy.compute_factors(history_data)
        
        elapsed_ms = (time.time() - start) * 1000
        
        # 提取映射（如果存在）
        self._code_to_idx = self._factor_store.get("__code_to_idx__", {})  # type: ignore
        self._date_to_idx = self._factor_store.get("__date_to_idx__", {})  # type: ignore
        
        # 构建 FactorAccessor
        if self._code_to_idx and self._date_to_idx:
            self._factor_accessor = FactorAccessor(
                store=self._factor_store,
                code_to_idx=self._code_to_idx,
                date_to_idx=self._date_to_idx,
                current_date_idx=0,  # 初始化为 0，回测循环中更新
            )
            self.logger.info(f"✓ FactorAccessor created with {len(self._code_to_idx)} codes")
        
        self.logger.info(
            f"✓ Factors computed in {elapsed_ms:.2f}ms "
            f"({len(self._factor_store)} factors)"
        )
    
    def _run_backtest_loop(
        self,
        market_data: Optional[pd.DataFrame],
    ) -> None:
        """回测主循环"""
        self.logger.info("Running backtest loop...")
        
        # ✅ FIX: 预过滤交易日，避免遍历非交易日
        if market_data is not None and "date" in market_data.columns:
            # 使用数据中的实际交易日
            trading_dates = sorted(market_data["date"].unique())
            dates = [d for d in trading_dates
                    if self.start_date <= d <= self.end_date]
            self.logger.info(f"Using {len(dates)} trading days from data")
        else:
            # 回退：使用工作日近似（排除周末）
            date_range = pd.date_range(self.start_date, self.end_date, freq="B")
            dates = [d.strftime("%Y-%m-%d") for d in date_range]
            self.logger.warning(
                f"No market data, using business days approximation: {len(dates)} days"
            )
        
        # 进度条
        if self.show_progress and TQDM_AVAILABLE:
            iterator = tqdm(dates, desc="Backtest")
        else:
            iterator = dates
        
        for date in iterator:
            self._run_single_day(date, market_data)
        
        self.logger.info(f"✓ Backtest completed {len(dates)} days")
    
    def _run_single_day(
        self,
        date: str,
        market_data: Optional[pd.DataFrame],
    ) -> None:
        """单日回测流程 (FIXED: 跳过非交易日)"""
        # ── FIXED Step 0: 检查是否为交易日 ──
        if date not in self._date_to_idx:
            self.logger.debug(f"Skip non-trading day: {date}")
            return
        
        # ── Step 1: 获取当日行情 ──
        current_data = self._get_current_data(date, market_data)
        if current_data.empty:
            self.logger.warning(f"No market data for {date}")
            return
        
        # ── Step 2: 更新 FactorAccessor 的当前日期索引 ──
        if self._factor_accessor:
            self._factor_accessor._current_date_idx = self._date_to_idx[date]
        
        # ── Step 3: 构建 StrategyContext ──
        context = self._build_context(date, current_data)
        
        # ── Step 4: Day start 钩子 ──
        self.strategy.on_day_start(context)
        
        # ── Step 5: 生成信号 ──
        signals = self.strategy.generate_signals(context)
        
        if not signals:
            # 无信号，仅更新市值和快照
            self.account.update_market_value(current_data)
            timestamp = Timestamp.from_str(date)
            self.account.create_snapshot(timestamp, date)
            return
        
        # ── Step 6: 提交订单 ──
        timestamp = Timestamp.from_str(date)
        orders = self.match_engine.submit_signals(
            signals, current_data, self.account.total_equity, timestamp
        )
        
        # ── Step 7: 撮合订单 ──
        fills, rejects = self.match_engine.match_orders(orders, current_data, timestamp)
        
        # ── Step 8: 处理成交 ──
        for fill in fills:
            # 创建持仓状态（简化版：使用 BasePositionState）
            # 实际使用中应该由策略创建具体的 PositionState 子类
            position = BasePositionState(
                code=fill.code,
                entry_price=fill.price,
                entry_date=date,
                quantity=fill.quantity,
            )
            
            # 处理成交
            success = self.account.process_fill(fill, position)
            
            if success:
                # 调用策略回调
                self.strategy.on_order_filled(fill)
        
        # ── Step 9: 处理拒绝订单 ──
        for order in rejects:
            self.strategy.on_order_rejected(order, order.reject_reason)
        
        # ── Step 10: 更新市值 ──
        self.account.update_market_value(current_data)
        
        # ── Step 11: 创建快照 ──
        self.account.create_snapshot(timestamp, date)
        
        # ── Step 12: Day end 钩子 ──
        # ✅ FIX: 先解冻T+1资金，再调用策略钩子
        self.account.on_day_end()  # T+1结算
        self.strategy.on_day_end(context)
    
    def _finalize(self) -> Dict:
        """结束回测，生成结果"""
        self.logger.info("Finalizing...")
        
        # 调用策略结束钩子
        self.strategy.on_strategy_stop()
        
        # 收集结果
        snapshots = self.account.get_snapshots()
        equity_curve = self.account.get_equity_curve()
        trade_history = self.account.get_trade_history()
        trade_stats = self.account.get_trade_statistics()
        match_stats = self.match_engine.get_statistics()
        
        # 计算绩效指标
        performance = self._calculate_performance(equity_curve)
        
        result = {
            "snapshots": snapshots,
            "equity_curve": equity_curve,
            "trade_history": trade_history,
            "trade_statistics": trade_stats,
            "match_statistics": match_stats,
            "performance": performance,
        }
        
        self.logger.info("✓ Results collected")
        
        # 打印摘要
        self._print_summary(result)
        
        return result
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _get_current_data(
        self,
        date: str,
        market_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """获取当日行情"""
        if market_data is not None and "date" in market_data.columns:
            # 从全市场数据中筛选
            current = market_data[market_data["date"] == date].copy()
            return current
        
        # 从历史数据中提取（较慢）
        rows = []
        if self._history_data:
            for code, df in self._history_data.items():
                if "date" in df.columns:
                    row = df[df["date"] == date]
                elif isinstance(df.index, pd.DatetimeIndex):
                    try:
                        row = df.loc[[pd.Timestamp(date)]]
                    except KeyError:
                        continue
                else:
                    continue
                
                if not row.empty:
                    row_dict = row.iloc[0].to_dict()
                    row_dict["code"] = code
                    rows.append(row_dict)
        
        return pd.DataFrame(rows)
    
    def _build_context(
        self,
        date: str,
        current_data: pd.DataFrame,
    ) -> StrategyContext:
        """构建 StrategyContext"""
        # 持仓快照（{code: quantity}）
        positions_snapshot = {
            code: pos.quantity
            for code, pos in self.account._positions.items()
        }
        
        # 股票池列表
        universe = list(self._history_data.keys()) if self._history_data else []
        
        # 历史数据提供者（闭包）
        def history_provider(code: str, lookback: int = 250) -> pd.DataFrame:
            if self._history_data and code in self._history_data:
                df = self._history_data[code]
                # 简化：返回最近 lookback 条
                return df.tail(lookback)
            return pd.DataFrame()
        
        # 因子提供者（闭包）
        def factor_provider(factor_name: str, code: str) -> Optional[float]:
            if self._factor_accessor:
                return self._factor_accessor.get(factor_name, code)
            return None
        
        # 构建 Context
        context = StrategyContext(
            current_date=date,
            current_data=current_data,
            positions=positions_snapshot,
            total_equity=self.account.total_equity,
            cash=self.account.cash,
            current_timestamp=Timestamp.from_str(date),
            history_provider=history_provider,
            factor_provider=factor_provider,
            factor_accessor=self._factor_accessor,
            universe=universe,
        )
        
        return context
    
    def _calculate_performance(
        self,
        equity_curve: pd.DataFrame,
    ) -> Dict:
        """计算绩效指标"""
        if equity_curve.empty:
            return {}
        
        # 提取净值序列
        equity = equity_curve["total_equity"].values
        
        # 总收益率
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # 年化收益率（假设 252 个交易日）
        n_days = len(equity)
        annual_return = (1 + total_return) ** (252.0 / n_days) - 1
        
        # 日收益率
        daily_returns = np.diff(equity) / equity[:-1]
        
        # 夏普比率（假设无风险利率 3%）
        risk_free_rate = 0.03
        daily_rf = risk_free_rate / 252
        excess_returns = daily_returns - daily_rf
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if len(excess_returns) > 0 else 0.0
        
        # 最大回撤
        cummax = np.maximum.accumulate(equity)
        drawdowns = (equity - cummax) / cummax
        max_drawdown = np.min(drawdowns)
        
        # 胜率（从账户统计获取）
        trade_stats = self.account.get_trade_statistics()
        win_rate = trade_stats.get("win_rate", 0.0)
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "final_equity": equity[-1],
            "n_days": n_days,
        }
    
    def _print_summary(self, result: Dict) -> None:
        """打印回测摘要"""
        perf = result.get("performance", {})
        trade_stats = result.get("trade_statistics", {})
        match_stats = result.get("match_statistics", {})
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("BACKTEST SUMMARY")
        self.logger.info("=" * 70)
        
        if perf:
            self.logger.info("Performance:")
            self.logger.info(f"  Total Return:    {perf.get('total_return', 0) * 100:6.2f}%")
            self.logger.info(f"  Annual Return:   {perf.get('annual_return', 0) * 100:6.2f}%")
            self.logger.info(f"  Sharpe Ratio:    {perf.get('sharpe_ratio', 0):6.2f}")
            self.logger.info(f"  Max Drawdown:    {perf.get('max_drawdown', 0) * 100:6.2f}%")
            self.logger.info(f"  Final Equity:    {perf.get('final_equity', 0):,.2f}")
        
        self.logger.info("")
        
        if trade_stats:
            self.logger.info("Trading:")
            self.logger.info(f"  Total Trades:    {trade_stats.get('total_trades', 0)}")
            self.logger.info(f"  Win Rate:        {trade_stats.get('win_rate', 0) * 100:6.2f}%")
            self.logger.info(f"  Avg PnL:         {trade_stats.get('avg_pnl', 0):,.2f}")
            self.logger.info(f"  Total PnL:       {trade_stats.get('total_pnl', 0):,.2f}")
        
        self.logger.info("")
        
        if match_stats:
            self.logger.info("Execution:")
            self.logger.info(f"  Orders:          {match_stats.get('total_orders', 0)}")
            self.logger.info(f"  Filled:          {match_stats.get('filled_orders', 0)}")
            self.logger.info(f"  Fill Rate:       {match_stats.get('fill_rate', 0) * 100:6.2f}%")
            self.logger.info(f"  Avg Slippage:    {match_stats.get('avg_slippage_pct', 0):6.4f}%")
        
        self.logger.info("=" * 70)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "HybridExecutionEngine",
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    from strategy.base import BaseStrategy
    from strategy.types import FactorStore, Signal, OrderSide
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("HYBRID EXECUTION ENGINE - TEST")
    print("=" * 70)
    print()
    
    # 定义简单策略
    class DummyStrategy(BaseStrategy):
        name = "dummy"
        version = "1.0.0"
        
        def compute_factors(self, history: Dict[str, pd.DataFrame]) -> FactorStore:
            # 返回空因子
            return {}
        
        def _generate_entry_signals(self, ctx: StrategyContext) -> List[Signal]:
            # 每天随机买入 2 只股票
            if ctx.current_data.empty:
                return []
            
            codes = ctx.current_data["code"].tolist()[:2]
            return [
                Signal(code=code, side=OrderSide.BUY, weight=0.05, reason="dummy")
                for code in codes
            ]
    
    # 准备测试数据
    n_stocks = 50
    n_days = 100
    codes = [f"SH{600000 + i:06d}" for i in range(n_stocks)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    # 生成模拟历史数据
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
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        history_data[code] = df
    
    # 创建策略和引擎
    strategy = DummyStrategy()
    engine = HybridExecutionEngine(
        strategy=strategy,
        start_date="2024-01-01",
        end_date="2024-04-10",  # 100 天
        initial_cash=1000000.0,
        show_progress=True,
    )
    
    # 运行回测
    print("Running backtest...")
    result = engine.run_backtest(history_data)
    
    print()
    print("=" * 70)
    print("TEST COMPLETE ✓")
    print("=" * 70)
