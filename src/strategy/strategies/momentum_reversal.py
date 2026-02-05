"""
Momentum-Reversal Combo Strategy
动量追踪 + 反转捕捉 双模式组合
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from ..base import BaseStrategy
from ..types import Signal, OrderSide, StrategyContext
from ...factors.alpha_engine import AlphaFactorEngine


class MomentumReversalStrategy(BaseStrategy):
    """
    动量-反转组合策略
    
    双模式:
    - 动量模式60%: RSRS>0.8 追强势
    - 反转模式40%: RSRS<-0.6 捕反弹
    """
    
    name = "momentum_reversal"
    version = "2.1.0"
    
    def __init__(
        self,
        momentum_n: int = 12,
        reversal_n: int = 8,
        momentum_threshold: float = 0.8,
        reversal_threshold: float = -0.6,
    ):
        super().__init__()
        self.momentum_n = momentum_n
        self.reversal_n = reversal_n
        self.momentum_threshold = momentum_threshold
        self.reversal_threshold = reversal_threshold
        
        self._position_modes: Dict[str, str] = {}
        self.logger = logging.getLogger(f"strategy.{self.name}")
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]):
        """计算因子"""
        engine = AlphaFactorEngine.from_dataframe_dict(history)
        return engine.compute()
    
    def _generate_entry_signals(self, context: StrategyContext) -> List[Signal]:
        """双模式入场"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        r2 = context.get_factor("rsrs_r2")
        
        if not rsrs:
            return signals
        
        momentum_candidates = []
        reversal_candidates = []
        
        for code in context.universe:
            rsrs_val = rsrs.get(code)
            r2_val = r2.get(code) if r2 else 1.0
            
            if not rsrs_val or r2_val < 0.7:
                continue
            
            # 动量模式
            if rsrs_val > self.momentum_threshold:
                momentum_candidates.append((code, rsrs_val))
            # 反转模式
            elif rsrs_val < self.reversal_threshold:
                reversal_candidates.append((code, abs(rsrs_val)))
        
        # 动量Top N (60%仓位)
        momentum_candidates.sort(key=lambda x: x[1], reverse=True)
        momentum_selected = momentum_candidates[:self.momentum_n]
        
        if momentum_selected:
            weight = 0.6 / len(momentum_selected)
            for code, score in momentum_selected:
                if code not in context.get_positions():
                    self._position_modes[code] = "momentum"
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.BUY,
                        weight=weight,
                        reason=f"动量{score:.2f}",
                    ))
        
        # 反转Top N (40%仓位)
        reversal_candidates.sort(key=lambda x: x[1], reverse=True)
        reversal_selected = reversal_candidates[:self.reversal_n]
        
        if reversal_selected:
            weight = 0.4 / len(reversal_selected)
            for code, score in reversal_selected:
                if code not in context.get_positions():
                    self._position_modes[code] = "reversal"
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.BUY,
                        weight=weight,
                        reason=f"反转{score:.2f}",
                    ))
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """双模式离场"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        current_prices = context.get_current_prices()
        
        for code, pos in context.get_positions().items():
            price = current_prices.get(code)
            if not price:
                continue
            
            mode = self._position_modes.get(code, "momentum")
            rsrs_val = rsrs.get(code) if rsrs else 0
            pnl = (price - pos.entry_price) / pos.entry_price
            
            if mode == "momentum":
                # 动量止损
                if rsrs_val < 0.3 or pnl < -0.04:
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.SELL,
                        weight=0.0,
                        reason=f"动量失效{rsrs_val:.2f}",
                    ))
            else:
                # 反转止盈/止损
                if rsrs_val > 0.5 or pnl > 0.08:
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.SELL,
                        weight=0.0,
                        reason=f"反转完成{pnl*100:.1f}%",
                    ))
                elif pnl < -0.06:
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.SELL,
                        weight=0.0,
                        reason=f"反转失败{pnl*100:.1f}%",
                    ))
        
        return signals
