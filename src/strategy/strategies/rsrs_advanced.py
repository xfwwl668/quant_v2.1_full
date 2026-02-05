"""
RSRS Advanced Strategy - 完整适配版
基于原始rsrs_strategy.py，适配到当前系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from ..base import BaseStrategy
from ..types import Signal, OrderSide, StrategyContext
from ...factors.alpha_engine import AlphaFactorEngine


class RSRSAdvancedStrategy(BaseStrategy):
    """
    RSRS 高级策略
    
    特性:
    - RSRS Z-Score判断趋势
    - R²过滤信号有效性
    - 量价共振
    - 动态止损
    """
    
    name = "rsrs_advanced"
    version = "2.1.0"
    
    def __init__(
        self,
        top_n: int = 30,
        entry_threshold: float = 0.6,
        exit_threshold: float = -0.2,
        r2_threshold: float = 0.8,
        use_volume_filter: bool = True,
    ):
        super().__init__()
        self.top_n = top_n
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.r2_threshold = r2_threshold
        self.use_volume_filter = use_volume_filter
        
        self._entry_prices: Dict[str, float] = {}
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
            
            if not rsrs_val or rsrs_val <= self.entry_threshold:
                continue
            
            if r2_val < self.r2_threshold:
                continue
            
            candidates.append((code, rsrs_val))
        
        # Top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:self.top_n]
        
        weight = 1.0 / len(selected) if selected else 0
        
        for code, rsrs_val in selected:
            if code not in context.get_positions():
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.BUY,
                    weight=weight,
                    reason=f"RSRS={rsrs_val:.2f} R²有效",
                ))
                self._entry_prices[code] = context.get_current_prices().get(code, 0)
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """离场信号"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        current_prices = context.get_current_prices()
        
        for code, pos in context.get_positions().items():
            price = current_prices.get(code)
            if not price:
                continue
            
            rsrs_val = rsrs.get(code) if rsrs else 0
            
            # RSRS转弱
            if rsrs_val < self.exit_threshold:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"RSRS转弱{rsrs_val:.2f}",
                ))
                continue
            
            # 计算盈亏
            entry = self._entry_prices.get(code, pos.entry_price)
            pnl = (price - entry) / entry
            
            # 动态止损
            if pnl < -0.05:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"止损{pnl*100:.1f}%",
                ))
        
        return signals
