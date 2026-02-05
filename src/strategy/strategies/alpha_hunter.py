"""
Alpha Hunter V2 Strategy - 完整适配版
高频短线策略，多层止损
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import logging

from ..base import BaseStrategy
from ..types import Signal, OrderSide, StrategyContext
from ...factors.alpha_engine import AlphaFactorEngine


@dataclass
class EnhancedPosition:
    """增强持仓状态"""
    code: str
    entry_price: float
    entry_date: str
    highest_price: float
    trailing_stop: float
    hard_stop: float
    lock_level: int = 0
    lock_levels: List[float] = None
    
    def __post_init__(self):
        if self.lock_levels is None:
            self.lock_levels = [0.03, 0.06, 0.09, 0.12, 0.15]
    
    def update_trailing_stop(self, current_price: float):
        """更新移动止损"""
        if current_price > self.highest_price:
            self.highest_price = current_price
        
        pnl = (current_price - self.entry_price) / self.entry_price
        
        # 每+3%利润，止损上移2%
        while (self.lock_level < len(self.lock_levels) and 
               pnl >= self.lock_levels[self.lock_level]):
            new_stop = self.entry_price * (1 + 0.02 * (self.lock_level + 1))
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
            self.lock_level += 1


class AlphaHunterStrategy(BaseStrategy):
    """
    Alpha Hunter V2 - 高频短线
    
    核心:
    - 自适应RSRS
    - 多层锁利止损
    - 快进快出
    """
    
    name = "alpha_hunter"
    version = "2.1.0"
    
    def __init__(
        self,
        top_n: int = 20,
        rsrs_threshold: float = 0.75,
        hard_stop: float = -0.04,
        lock_levels: List[float] = None,
    ):
        super().__init__()
        self.top_n = top_n
        self.rsrs_threshold = rsrs_threshold
        self.hard_stop = hard_stop
        self.lock_levels = lock_levels or [0.03, 0.06, 0.09, 0.12]
        
        self._positions: Dict[str, EnhancedPosition] = {}
        self.logger = logging.getLogger(f"strategy.{self.name}")
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]):
        """计算因子"""
        engine = AlphaFactorEngine.from_dataframe_dict(history)
        factors = engine.compute()
        return factors
    
    def _generate_entry_signals(self, context: StrategyContext) -> List[Signal]:
        """入场信号"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        r2 = context.get_factor("rsrs_r2")
        
        if not rsrs:
            return signals
        
        candidates = []
        for code in context.universe:
            rsrs_val = rsrs.get(code)
            r2_val = r2.get(code) if r2 else 1.0
            
            if rsrs_val and rsrs_val > self.rsrs_threshold and r2_val > 0.8:
                candidates.append((code, rsrs_val))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:self.top_n]
        
        weight = 1.0 / len(selected) if selected else 0
        
        for code, rsrs_val in selected:
            if code not in context.get_positions():
                price = context.get_current_prices().get(code, 0)
                
                # 创建增强持仓记录
                self._positions[code] = EnhancedPosition(
                    code=code,
                    entry_price=price,
                    entry_date=context.current_date,
                    highest_price=price,
                    trailing_stop=price * (1 + self.hard_stop),
                    hard_stop=price * (1 + self.hard_stop),
                    lock_levels=self.lock_levels,
                )
                
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.BUY,
                    weight=weight,
                    reason=f"AlphaHunter RSRS={rsrs_val:.2f}",
                ))
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """多层止损"""
        signals = []
        
        current_prices = context.get_current_prices()
        
        for code, pos_state in list(self._positions.items()):
            if code not in context.get_positions():
                continue
            
            price = current_prices.get(code)
            if not price:
                continue
            
            # 更新止损线
            pos_state.update_trailing_stop(price)
            
            pnl = (price - pos_state.entry_price) / pos_state.entry_price
            
            # 硬止损
            if pnl < self.hard_stop:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"硬止损{pnl*100:.1f}%",
                ))
                del self._positions[code]
                continue
            
            # 移动止损
            if price < pos_state.trailing_stop:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"锁利{pnl*100:.1f}% L{pos_state.lock_level}",
                ))
                del self._positions[code]
        
        return signals
