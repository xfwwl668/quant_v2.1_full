"""
Sentiment Reversal Strategy
基于极端情绪的反转策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from ..base import BaseStrategy
from ..types import Signal, OrderSide, StrategyContext
from ...factors.alpha_engine import AlphaFactorEngine


class SentimentReversalStrategy(BaseStrategy):
    """
    情绪反转策略
    
    捕捉超卖后的反弹
    - 入场: RSRS<-0.8 (极端超卖)
    - 离场: RSRS>0 (回归均值) 或 止盈12%
    """
    
    name = "sentiment_reversal"
    version = "2.1.0"
    
    def __init__(
        self,
        top_n: int = 12,
        oversold_threshold: float = -0.8,
        exit_threshold: float = 0.0,
        max_stop_loss: float = -0.08,
        target_profit: float = 0.12,
    ):
        super().__init__()
        self.top_n = top_n
        self.oversold_threshold = oversold_threshold
        self.exit_threshold = exit_threshold
        self.max_stop_loss = max_stop_loss
        self.target_profit = target_profit
        
        self.logger = logging.getLogger(f"strategy.{self.name}")
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]):
        """计算因子"""
        high, low, close, open_arr, volume, codes, dates = (
            AlphaFactorEngine.from_dataframe_dict(history)
        )
        engine = AlphaFactorEngine()
        factors, code_to_idx, date_to_idx = engine.compute(
            high,
            low,
            close,
            open_arr,
            volume,
            codes,
            dates,
            rsrs_window=18,
            zscore_window=600,
        )
        factors["__code_to_idx__"] = code_to_idx
        factors["__date_to_idx__"] = date_to_idx
        factor_count = len([name for name in factors.keys() if not name.startswith("__")])
        self.logger.info(f"✓ 因子计算完成: {factor_count} 个因子")
        return factors
    
    def _generate_entry_signals(self, context: StrategyContext) -> List[Signal]:
        """捕捉超卖"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        r2 = context.get_factor("rsrs_r2")
        
        if not rsrs:
            return signals
        
        oversold = []
        for code in context.universe:
            rsrs_val = rsrs.get(code)
            r2_val = r2.get(code) if r2 else 1.0
            
            # 超卖且信号可靠
            if rsrs_val and rsrs_val < self.oversold_threshold and r2_val > 0.75:
                oversold.append((code, rsrs_val))
        
        # 选最超卖的
        oversold.sort(key=lambda x: x[1])
        selected = oversold[:self.top_n]
        
        weight = 1.0 / len(selected) if selected else 0
        
        for code, rsrs_val in selected:
            if code not in context.get_positions():
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.BUY,
                    weight=weight,
                    reason=f"超卖{rsrs_val:.2f}",
                ))
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """反转完成或止损"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        current_prices = context.get_current_prices()
        
        for code, pos in context.get_positions().items():
            price = current_prices.get(code)
            if not price:
                continue
            
            rsrs_val = rsrs.get(code) if rsrs else 0
            pnl = (price - pos.entry_price) / pos.entry_price
            
            # 止盈
            if pnl > self.target_profit:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"目标{pnl*100:.1f}%",
                ))
            elif rsrs_val > self.exit_threshold:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"反转完成{rsrs_val:.2f}",
                ))
            # 止损
            elif pnl < self.max_stop_loss:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"止损{pnl*100:.1f}%",
                ))
        
        return signals
