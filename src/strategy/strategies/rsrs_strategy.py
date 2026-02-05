"""
src/strategy/strategies/rsrs_strategy.py
========================================
RSRS 动量策略实现（v2.0.1 生产级）

策略逻辑：
  1. 使用 RSRS 因子（Resistance Support Relative Strength）
  2. Top-N 选股（adaptive RSRS 得分最高的 N 只）
  3. 等权配置 + 每日调仓
  4. 止损保护（-5% 硬止损）

契约对齐：
  - PositionState 使用 BasePositionState（__slots__ 契约）
  - Signal/Order/Fill 严格对齐 types.py 定义
  - compute_factors() 返回 FactorStore 字典
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import BaseStrategy
from ..types import (
    BasePositionState,
    FactorStore,
    Order,
    OrderSide,
    OrderType,
    Signal,
    StrategyContext,
    Timestamp,
)


# ============================================================================
# RSRS 动量策略
# ============================================================================

class RSRSMomentumStrategy(BaseStrategy):
    """
    RSRS 动量策略（生产级实现）。
    
    参数：
        top_n: 选股数量（默认 50）
        lookback: 回看窗口（默认 250）
        rsrs_threshold: RSRS 阈值（默认 0.5）
        stop_loss_pct: 止损比例（默认 -0.05）
        enable_stop_loss: 是否启用止损（默认 True）
    
    因子使用：
        - rsrs_adaptive: 自适应 RSRS 得分
        - rsrs_valid: 有效性标记（r² > 0.8）
        - rsrs_r2: 拟合优度
    
    契约保证：
        - PositionState: BasePositionState（__slots__）
        - FactorStore: Dict[str, pd.Series]
        - 所有 Signal/Order 对齐 types.py
    """
    
    name = "rsrs_momentum"
    
    def __init__(
        self,
        top_n: int = 50,
        lookback: int = 250,
        rsrs_threshold: float = 0.5,
        stop_loss_pct: float = -0.05,
        enable_stop_loss: bool = True,
    ):
        super().__init__()
        
        # 策略参数
        self.top_n = top_n
        self.lookback = lookback
        self.rsrs_threshold = rsrs_threshold
        self.stop_loss_pct = stop_loss_pct
        self.enable_stop_loss = enable_stop_loss
        
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> FactorStore:
        """
        计算 RSRS 因子（契约方法）。
        
        Args:
            history: {code: DataFrame} 历史数据
        
        Returns:
            FactorStore: {factor_name: pd.Series}
        
        契约：
            - 必须返回 Dict[str, pd.Series]
            - Series.index 必须是日期（datetime64）
            - Series.values 必须是 float64
        """
        from ...factors.alpha_engine import AlphaFactorEngine
        
        # 使用 AlphaFactorEngine 计算因子
        try:
            engine = AlphaFactorEngine.from_dataframe_dict(
                history,
                rsrs_window=18,
                zscore_window=600
            )
            
            factors = engine.compute()
            
            self.logger.info(
                f"✓ 因子计算完成: {len(factors)} 个因子 × "
                f"{engine.n_stocks} 股票 × {engine.n_days} 天"
            )
            
            return factors
        
        except Exception as e:
            self.logger.error(f"因子计算失败: {e}")
            raise
    
    def _generate_entry_signals(self, context: StrategyContext) -> List[Signal]:
        """
        生成入场信号（Top-N 选股）。
        
        逻辑：
          1. 获取最新 rsrs_adaptive 得分
          2. 过滤有效信号（rsrs_valid == 1）
          3. 选择 Top-N
          4. 生成等权配置信号
        
        契约：
          - 返回 List[Signal]
          - Signal 必须对齐 types.py 定义
        """
        signals: List[Signal] = []
        
        # 获取当前日期
        current_date = context.current_date
        
        # 获取 RSRS 因子
        rsrs_adaptive = context.get_factor("rsrs_adaptive")
        rsrs_valid = context.get_factor("rsrs_valid")
        
        if rsrs_adaptive is None or rsrs_valid is None:
            self.logger.warning("RSRS 因子缺失，跳过信号生成")
            return signals
        
        # 提取最新值
        latest_scores: Dict[str, float] = {}
        
        for code in context.universe:
            score = rsrs_adaptive.get(code)
            valid = rsrs_valid.get(code)
            
            # 过滤无效信号
            if score is None or valid is None:
                continue
            
            if np.isnan(score) or np.isnan(valid):
                continue
            
            if valid < 0.5:  # rsrs_valid == 0
                continue
            
            if score < self.rsrs_threshold:
                continue
            
            latest_scores[code] = score
        
        if not latest_scores:
            self.logger.debug("无有效 RSRS 信号")
            return signals
        
        # Top-N 选股
        sorted_stocks = sorted(
            latest_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_n]
        
        # 等权配置
        target_weight = 1.0 / len(sorted_stocks)
        
        for code, score in sorted_stocks:
            signals.append(Signal(
                code=code,
                direction=OrderSide.BUY,
                weight=target_weight,
                reason=f"RSRS={score:.2f}",
                timestamp=Timestamp.from_str(current_date)
            ))
        
        self.logger.debug(
            f"生成 {len(signals)} 个入场信号 | Top-N={self.top_n} | "
            f"最高得分={sorted_stocks[0][1]:.2f}"
        )
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """
        生成出场信号（止损 + 主动调仓）。
        
        逻辑：
          1. 检查止损（持仓亏损 > stop_loss_pct）
          2. 检查 RSRS 失效（rsrs_valid == 0）
          3. 检查持仓不在 Top-N（主动调仓）
        
        契约：
          - 返回 List[Signal]
          - Signal.direction == SELL
        """
        signals: List[Signal] = []
        
        # 获取当前持仓
        positions = context.get_positions()
        if not positions:
            return signals
        
        # 获取当前价格
        current_prices = context.get_current_prices()
        
        # 获取 RSRS 因子
        rsrs_adaptive = context.get_factor("rsrs_adaptive")
        rsrs_valid = context.get_factor("rsrs_valid")
        
        for code, pos in positions.items():
            current_price = current_prices.get(code)
            if current_price is None or current_price <= 0:
                continue
            
            # 检查止损
            if self.enable_stop_loss:
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                
                if pnl_pct <= self.stop_loss_pct:
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.SELL,
                        weight=0.0,
                        reason=f"止损: {pnl_pct*100:.1f}%",
                        timestamp=Timestamp.from_str(context.current_date)
                    ))
                    continue
            
            # 检查 RSRS 失效
            if rsrs_valid is not None:
                valid = rsrs_valid.get(code)
                if valid is not None and valid < 0.5:
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.SELL,
                        weight=0.0,
                        reason="RSRS 失效",
                        timestamp=Timestamp.from_str(context.current_date)
                    ))
                    continue
            
            # 检查是否还在 Top-N（主动调仓）
            if rsrs_adaptive is not None:
                score = rsrs_adaptive.get(code)
                
                # 获取当前 Top-N
                latest_scores: Dict[str, float] = {}
                for c in context.universe:
                    s = rsrs_adaptive.get(c)
                    v = rsrs_valid.get(c) if rsrs_valid else None
                    
                    if s is not None and not np.isnan(s):
                        if v is None or v >= 0.5:
                            latest_scores[c] = s
                
                if latest_scores:
                    sorted_stocks = sorted(
                        latest_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:self.top_n]
                    
                    top_codes = {c for c, _ in sorted_stocks}
                    
                    if code not in top_codes:
                        signals.append(Signal(
                            code=code,
                            direction=OrderSide.SELL,
                            weight=0.0,
                            reason=f"退出 Top-{self.top_n}",
                            timestamp=Timestamp.from_str(context.current_date)
                        ))
        
        if signals:
            self.logger.debug(f"生成 {len(signals)} 个出场信号")
        
        return signals
    
    def on_trade(
        self,
        code: str,
        side: OrderSide,
        price: float,
        quantity: int,
        timestamp: Timestamp
    ) -> None:
        """交易回调（可选实现）"""
        self.logger.info(
            f"交易执行: {code} | {side.value} | "
            f"{quantity} 股 @ {price:.2f}"
        )
    
    def on_fill(
        self,
        code: str,
        side: OrderSide,
        filled_quantity: int,
        filled_price: float,
        timestamp: Timestamp
    ) -> None:
        """成交回调（可选实现）"""
        pass
    
    def create_position_state(
        self,
        code: str,
        entry_price: float,
        entry_date: str,
        quantity: int
    ) -> BasePositionState:
        """
        创建持仓状态（契约方法）。
        
        契约：
          - 必须返回 BasePositionState 或其子类
          - 必须使用 __slots__
        """
        return BasePositionState(
            code=code,
            entry_price=entry_price,
            entry_date=entry_date,
            quantity=quantity,
            stop_loss_price=entry_price * (1 + self.stop_loss_pct) if self.enable_stop_loss else 0.0,
            take_profit_price=0.0,
            trailing_stop_price=0.0,
            highest_price=entry_price,
            lowest_price=entry_price,
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = ['RSRSMomentumStrategy']
