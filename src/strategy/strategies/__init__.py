"""
策略集合 v2.1.0 - 完整版
包含6个生产级量化策略
"""

from .rsrs_strategy import RSRSMomentumStrategy
from .rsrs_advanced import RSRSAdvancedStrategy
from .alpha_hunter import AlphaHunterStrategy
from .short_term import ShortTermStrategy
from .momentum_reversal import MomentumReversalStrategy
from .sentiment_reversal import SentimentReversalStrategy

__all__ = [
    'RSRSMomentumStrategy',      # 基础RSRS动量
    'RSRSAdvancedStrategy',       # RSRS高级版
    'AlphaHunterStrategy',        # Alpha Hunter高频
    'ShortTermStrategy',          # 短线高胜率
    'MomentumReversalStrategy',   # 动量反转组合
    'SentimentReversalStrategy',  # 情绪反转
]
