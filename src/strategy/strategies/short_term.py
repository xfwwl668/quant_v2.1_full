"""
Short-Term RSRS Strategy - 高胜率短线
严格过滤 + 时间止损
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
import logging

from ..base import BaseStrategy
from ..types import Signal, OrderSide, StrategyContext
from ...factors.alpha_engine import AlphaFactorEngine


class ShortTermStrategy(BaseStrategy):
    """
    高胜率短线RSRS
    
    入场: RSRS>0.7, R²>0.8
    离场: RSRS<0.3 或 止损-3% 或 时间5天
    """
    
    name = "short_term"
    version = "2.1.0"
    
    def __init__(
        self,
        top_n: int = 15,
        rsrs_entry: float = 0.7,
        rsrs_exit: float = 0.3,
        stop_loss: float = -0.03,
        max_holding_days: int = 5,
    ):
        super().__init__()
        self.top_n = top_n
        self.rsrs_entry = rsrs_entry
        self.rsrs_exit = rsrs_exit
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days
        
        self._entry_dates: Dict[str, str] = {}
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
        """严格过滤入场"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        r2 = context.get_factor("rsrs_r2")
        
        if not rsrs:
            return signals
        
        candidates = []
        for code in context.universe:
            rsrs_val = rsrs.get(code)
            r2_val = r2.get(code) if r2 else 1.0
            
            # 严格过滤
            if rsrs_val and rsrs_val > self.rsrs_entry and r2_val > 0.8:
                candidates.append((code, rsrs_val))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:self.top_n]
        
        weight = 1.0 / len(selected) if selected else 0
        
        for code, score in selected:
            if code not in context.get_positions():
                self._entry_dates[code] = context.current_date
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.BUY,
                    weight=weight,
                    reason=f"短线入场{score:.2f}",
                ))
        
        return signals
    
    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """多重止损"""
        signals = []
        
        rsrs = context.get_factor("rsrs_adaptive")
        current_prices = context.get_current_prices()
        
        for code, pos in context.get_positions().items():
            price = current_prices.get(code)
            if not price:
                continue
            
            pnl = (price - pos.entry_price) / pos.entry_price
            rsrs_val = rsrs.get(code) if rsrs else 0
            
            # 止损1: 固定止损
            if pnl < self.stop_loss:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"止损{pnl*100:.1f}%",
                ))
                continue
            
            # 止损2: RSRS转弱
            if rsrs_val < self.rsrs_exit:
                signals.append(Signal(
                    code=code,
                    direction=OrderSide.SELL,
                    weight=0.0,
                    reason=f"RSRS转弱{rsrs_val:.2f}",
                ))
                continue
            
            # 止损3: 时间止损
            entry_date = self._entry_dates.get(code)
            if entry_date:
                days = self._calc_days(entry_date, context.current_date)
                if days >= self.max_holding_days and pnl < 0:
                    signals.append(Signal(
                        code=code,
                        direction=OrderSide.SELL,
                        weight=0.0,
                        reason=f"时间止损{days}天",
                    ))
        
        return signals
    
    def _calc_days(self, start: str, end: str) -> int:
        """计算天数"""
        try:
            d1 = datetime.strptime(start, "%Y-%m-%d")
            d2 = datetime.strptime(end, "%Y-%m-%d")
            return (d2 - d1).days
        except Exception as e:
            return 0
