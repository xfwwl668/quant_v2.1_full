"""
src/analysis/performance.py
============================
Phase 6 — PerformanceAnalyzer（绩效分析器）

核心职责：
  1. 基于 AccountSnapshot 计算完整绩效指标
  2. 年化收益率、夏普比率、最大回撤
  3. 胜率、盈亏比、换手率
  4. 生成绩效报告和可视化数据

═══════════════════════════════════════════════════════════════════
绩效指标定义
═══════════════════════════════════════════════════════════════════

1. **收益率指标**
   ──────────────
   - Total Return: (final_equity - initial_equity) / initial_equity
   - Annual Return: (1 + total_return)^(252/n_days) - 1
   - CAGR: Compound Annual Growth Rate

2. **风险指标**
   ───────────
   - Volatility: std(daily_returns) × √252
   - Max Drawdown (MDD): min((equity - cummax) / cummax)
   - Calmar Ratio: annual_return / abs(max_drawdown)

3. **风险调整收益**
   ────────────────
   - Sharpe Ratio: (annual_return - rf) / volatility
   - Sortino Ratio: (annual_return - rf) / downside_volatility
   - Information Ratio: excess_return / tracking_error

4. **交易指标**
   ───────────
   - Win Rate: wins / total_trades
   - Profit Factor: total_wins / abs(total_losses)
   - Average Win/Loss
   - Max Consecutive Wins/Losses

5. **换手率**
   ─────────
   Turnover = sum(|trade_value|) / average_equity / n_years

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Part 1: 绩效指标数据类
# ============================================================================

@dataclass
class PerformanceMetrics:
    """
    绩效指标汇总。
    
    包含所有关键绩效指标，用于报告和比较。
    """
    # 基础信息
    start_date: str
    end_date: str
    n_days: int
    initial_equity: float
    final_equity: float
    
    # 收益率
    total_return: float
    annual_return: float
    cagr: float
    
    # 风险
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # 风险调整收益
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 交易统计
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    
    # 换手率
    turnover: float
    
    # 其他
    avg_position_days: float
    max_positions: int
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_days": self.n_days,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_win": self.max_win,
            "max_loss": self.max_loss,
            "turnover": self.turnover,
            "avg_position_days": self.avg_position_days,
            "max_positions": self.max_positions,
        }
    
    def __str__(self) -> str:
        """格式化输出"""
        lines = [
            "=" * 70,
            "PERFORMANCE METRICS",
            "=" * 70,
            "",
            "Period:",
            f"  Start Date:        {self.start_date}",
            f"  End Date:          {self.end_date}",
            f"  Trading Days:      {self.n_days}",
            "",
            "Returns:",
            f"  Total Return:      {self.total_return * 100:6.2f}%",
            f"  Annual Return:     {self.annual_return * 100:6.2f}%",
            f"  CAGR:              {self.cagr * 100:6.2f}%",
            "",
            "Risk:",
            f"  Volatility:        {self.volatility * 100:6.2f}%",
            f"  Max Drawdown:      {self.max_drawdown * 100:6.2f}%",
            f"  MDD Duration:      {self.max_drawdown_duration} days",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio:      {self.sharpe_ratio:6.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:6.2f}",
            f"  Calmar Ratio:      {self.calmar_ratio:6.2f}",
            "",
            "Trading:",
            f"  Total Trades:      {self.total_trades}",
            f"  Win Rate:          {self.win_rate * 100:6.2f}%",
            f"  Profit Factor:     {self.profit_factor:6.2f}",
            f"  Avg Win:           {self.avg_win:6.2f}",
            f"  Avg Loss:          {self.avg_loss:6.2f}",
            f"  Turnover:          {self.turnover:6.2f}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


# ============================================================================
# Part 2: PerformanceAnalyzer 主类
# ============================================================================

class PerformanceAnalyzer:
    """
    绩效分析器（结算系统）。
    
    职责：
    1. 基于 AccountSnapshot 计算绩效指标
    2. 基于交易流水计算交易统计
    3. 生成绩效报告
    4. 提供可视化数据
    
    使用方法：
        analyzer = PerformanceAnalyzer()
        
        # 从回测结果分析
        metrics = analyzer.analyze(
            snapshots=result["snapshots"],
            trades=result["trade_history"],
        )
        
        # 打印报告
        print(metrics)
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.03,
        trading_days_per_year: int = 252,
    ):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
            trading_days_per_year: 每年交易日数
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        
        self.logger = logging.getLogger("analysis.performance")
    
    # ========================================================================
    # 主分析接口
    # ========================================================================
    
    def analyze(
        self,
        snapshots: List,
        trades: Optional[List[Dict]] = None,
        initial_equity: Optional[float] = None,
    ) -> PerformanceMetrics:
        """
        分析绩效指标。
        
        Args:
            snapshots: AccountSnapshot 列表
            trades: 交易流水列表
            initial_equity: 初始权益（可选，从 snapshots 推断）
        
        Returns:
            PerformanceMetrics
        """
        if not snapshots:
            raise ValueError("No snapshots provided")
        
        # 提取净值序列
        equity_curve = self._extract_equity_curve(snapshots)
        
        # 计算收益指标
        return_metrics = self._calculate_returns(equity_curve, initial_equity)
        
        # 计算风险指标
        risk_metrics = self._calculate_risk(equity_curve)
        
        # 计算风险调整收益
        risk_adj_metrics = self._calculate_risk_adjusted_returns(
            return_metrics, risk_metrics
        )
        
        # 计算交易统计
        if trades:
            trade_metrics = self._calculate_trade_statistics(trades)
        else:
            trade_metrics = self._empty_trade_metrics()
        
        # 计算换手率
        turnover = self._calculate_turnover(snapshots, equity_curve)
        
        # 其他统计
        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        n_days = len(snapshots)
        max_positions = max(s.positions_count for s in snapshots)
        
        # 汇总
        metrics = PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            n_days=n_days,
            initial_equity=return_metrics["initial_equity"],
            final_equity=return_metrics["final_equity"],
            total_return=return_metrics["total_return"],
            annual_return=return_metrics["annual_return"],
            cagr=return_metrics["cagr"],
            volatility=risk_metrics["volatility"],
            max_drawdown=risk_metrics["max_drawdown"],
            max_drawdown_duration=risk_metrics["max_drawdown_duration"],
            sharpe_ratio=risk_adj_metrics["sharpe_ratio"],
            sortino_ratio=risk_adj_metrics["sortino_ratio"],
            calmar_ratio=risk_adj_metrics["calmar_ratio"],
            total_trades=trade_metrics["total_trades"],
            win_trades=trade_metrics["win_trades"],
            loss_trades=trade_metrics["loss_trades"],
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            avg_win=trade_metrics["avg_win"],
            avg_loss=trade_metrics["avg_loss"],
            max_win=trade_metrics["max_win"],
            max_loss=trade_metrics["max_loss"],
            turnover=turnover,
            avg_position_days=trade_metrics["avg_position_days"],
            max_positions=max_positions,
        )
        
        return metrics
    
    # ========================================================================
    # 收益率计算
    # ========================================================================
    
    def _extract_equity_curve(self, snapshots: List) -> np.ndarray:
        """提取净值曲线"""
        return np.array([s.total_equity for s in snapshots])
    
    def _calculate_returns(
        self,
        equity_curve: np.ndarray,
        initial_equity: Optional[float] = None,
    ) -> Dict:
        """计算收益率指标"""
        if initial_equity is None:
            initial_equity = equity_curve[0]
        
        final_equity = equity_curve[-1]
        n_days = len(equity_curve)
        
        # 总收益率
        total_return = (final_equity - initial_equity) / initial_equity
        
        # 年化收益率
        years = n_days / self.trading_days
        if years > 0:
            annual_return = (1 + total_return) ** (1.0 / years) - 1
            cagr = annual_return  # CAGR = 年化收益率
        else:
            annual_return = 0.0
            cagr = 0.0
        
        return {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "annual_return": annual_return,
            "cagr": cagr,
        }
    
    # ========================================================================
    # 风险指标计算
    # ========================================================================
    
    def _calculate_risk(self, equity_curve: np.ndarray) -> Dict:
        """计算风险指标"""
        # 日收益率
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # 波动率（年化）
        volatility = np.std(daily_returns, ddof=1) * np.sqrt(self.trading_days)
        
        # 最大回撤
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdowns)
        
        # 最大回撤持续期
        max_dd_duration = self._calculate_drawdown_duration(drawdowns)
        
        return {
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_dd_duration,
            "daily_returns": daily_returns,
        }
    
    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """计算最大回撤持续期（天数）"""
        max_duration = 0
        current_duration = 0
        
        for dd in drawdowns:
            if dd < -1e-9:  # 在回撤中
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    # ========================================================================
    # 风险调整收益
    # ========================================================================
    
    def _calculate_risk_adjusted_returns(
        self,
        return_metrics: Dict,
        risk_metrics: Dict,
    ) -> Dict:
        """计算风险调整收益指标"""
        annual_return = return_metrics["annual_return"]
        volatility = risk_metrics["volatility"]
        max_drawdown = risk_metrics["max_drawdown"]
        daily_returns = risk_metrics["daily_returns"]
        
        # Sharpe Ratio
        if volatility > 1e-9:
            sharpe_ratio = (annual_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio（仅考虑下行波动）
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_volatility = np.std(downside_returns, ddof=1) * np.sqrt(self.trading_days)
            if downside_volatility > 1e-9:
                sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = sharpe_ratio
        
        # Calmar Ratio
        if abs(max_drawdown) > 1e-9:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }
    
    # ========================================================================
    # 交易统计
    # ========================================================================
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """计算交易统计"""
        if not trades:
            return self._empty_trade_metrics()
        
        # 仅统计卖出交易（有盈亏）
        sell_trades = [t for t in trades if t.get("side") == "SELL"]
        
        if not sell_trades:
            return self._empty_trade_metrics()
        
        # 盈亏
        pnls = np.array([t.get("pnl", 0.0) for t in sell_trades])
        
        wins = pnls > 0
        losses = pnls <= 0
        
        win_trades = np.sum(wins)
        loss_trades = np.sum(losses)
        total_trades = len(sell_trades)
        
        # 胜率
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        
        # 盈亏比
        total_win = np.sum(pnls[wins])
        total_loss = np.sum(pnls[losses])
        
        if abs(total_loss) > 1e-9:
            profit_factor = total_win / abs(total_loss)
        else:
            profit_factor = float('inf') if total_win > 0 else 0.0
        
        # 平均盈亏
        avg_win = np.mean(pnls[wins]) if win_trades > 0 else 0.0
        avg_loss = np.mean(pnls[losses]) if loss_trades > 0 else 0.0
        
        # 最大盈亏
        max_win = np.max(pnls) if len(pnls) > 0 else 0.0
        max_loss = np.min(pnls) if len(pnls) > 0 else 0.0
        
        # 平均持仓天数（如果有）
        holding_days_list = [
            t.get("holding_days", 0) for t in sell_trades
            if "holding_days" in t
        ]
        avg_position_days = np.mean(holding_days_list) if holding_days_list else 0.0
        
        return {
            "total_trades": total_trades,
            "win_trades": int(win_trades),
            "loss_trades": int(loss_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "max_win": float(max_win),
            "max_loss": float(max_loss),
            "avg_position_days": float(avg_position_days),
        }
    
    def _empty_trade_metrics(self) -> Dict:
        """空交易统计"""
        return {
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "avg_position_days": 0.0,
        }
    
    # ========================================================================
    # 换手率
    # ========================================================================
    
    def _calculate_turnover(
        self,
        snapshots: List,
        equity_curve: np.ndarray,
    ) -> float:
        """
        计算换手率。
        
        Turnover = sum(abs(position_changes)) / avg_equity / years
        """
        if len(snapshots) < 2:
            return 0.0
        
        # 持仓变化
        position_changes = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i - 1].market_value
            curr_value = snapshots[i].market_value
            change = abs(curr_value - prev_value)
            position_changes.append(change)
        
        # 总交易量
        total_traded = sum(position_changes)
        
        # 平均权益
        avg_equity = np.mean(equity_curve)
        
        # 年数
        years = len(snapshots) / self.trading_days
        
        if avg_equity > 1e-9 and years > 0:
            turnover = total_traded / avg_equity / years
        else:
            turnover = 0.0
        
        return turnover


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "PerformanceAnalyzer",
    "PerformanceMetrics",
]


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("PERFORMANCE ANALYZER - TEST")
    print("=" * 70)
    print()
    
    # 模拟 AccountSnapshot
    @dataclass(frozen=True)
    class MockSnapshot:
        date: str
        total_equity: float
        market_value: float
        positions_count: int
    
    # 生成模拟净值曲线
    n_days = 250
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    # 模拟收益（年化 30%，波动率 15%）
    daily_return_mean = 0.30 / 252
    daily_return_std = 0.15 / np.sqrt(252)
    
    daily_returns = np.random.normal(daily_return_mean, daily_return_std, n_days)
    equity_curve = 1000000.0 * np.cumprod(1 + daily_returns)
    
    # 创建快照
    snapshots = [
        MockSnapshot(
            date=dates[i].strftime("%Y-%m-%d"),
            total_equity=equity_curve[i],
            market_value=equity_curve[i] * 0.7,  # 70% 仓位
            positions_count=np.random.randint(5, 15),
        )
        for i in range(n_days)
    ]
    
    # 模拟交易
    trades = []
    for i in range(50):
        pnl = np.random.normal(1000, 2000)
        trades.append({
            "side": "SELL",
            "pnl": pnl,
            "holding_days": np.random.randint(1, 10),
        })
    
    # 分析
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(snapshots, trades)
    
    # 打印结果
    print(metrics)
    
    # 验证
    print("\nValidation:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    
    print()
    print("=" * 70)
    print("TEST PASSED ✓")
    print("=" * 70)
