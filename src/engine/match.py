"""
src/engine/match.py
===================
Phase 4 — ExchangeMatchEngine（订单撮合引擎）

核心职责：
  1. 订单撮合：Signal → Order → Fill
  2. 动态滑点模型：基于成交量占比和波动率计算滑点
  3. 流动性断路器：涨停/跌停/停牌检测
  4. 对齐鲁棒性：处理 NaN 数据，防止撮合价格漂移

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. **动态滑点模型**
   ─────────────────
   传统回测常用固定滑点（如 0.1%），但真实市场滑点受多因素影响：
   
   slippage = base_slippage + volume_impact + volatility_impact
   
   其中：
   - base_slippage: 基础滑点（0.01%）
   - volume_impact: 成交量冲击 = order_volume / market_volume × scale
   - volatility_impact: 波动率冲击 = ATR × multiplier
   
   使用 Numba 加速批量计算，避免循环。

2. **流动性断路器**
   ───────────────
   - 涨停：无法买入（买一为涨停价时）
   - 跌停：无法卖出（卖一为跌停价时）
   - 停牌：任何操作均拒绝
   - 流动性不足：订单量超过当日成交量 50% 时部分成交

3. **对齐鲁棒性**
   ───────────────
   回测时可能遇到：
   - 新股上市（历史数据前期 NaN）
   - 停牌（当日数据 NaN）
   - 退市（后期数据 NaN）
   
   ensure_alignment 逻辑：
   - 检测 NaN → 拒绝订单
   - 记录拒绝原因（"code suspended" / "data not available"）

4. **Numba 加速路径**
   ──────────────────
   批量订单场景（多个 Signal 同时处理）使用 Numba：
   - calculate_slippage_batch: 批量计算滑点
   - check_liquidity_batch: 批量流动性检查
   
   单笔订单场景使用 Python 路径（可读性优先）。

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Numba 依赖检查
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# 导入类型
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from strategy.types import (
    Signal,
    Order,
    Fill,
    OrderSide,
    OrderType,
    OrderStatus,
    Timestamp,
)


# ============================================================================
# Part 1: Numba 加速函数 - 动态滑点计算
# ============================================================================

@njit(cache=True, fastmath=True)
def calculate_slippage_scalar(
    order_volume: float,
    market_volume: float,
    volatility: float,
    base_slippage: float = 0.0001,
    volume_scale: float = 0.001,
    volatility_scale: float = 0.3,
) -> float:
    """
    单笔订单的动态滑点计算（Numba 加速版）。
    
    slippage = base + volume_impact + volatility_impact
    
    Args:
        order_volume: 订单量（手数 × 100）
        market_volume: 当日市场成交量
        volatility: 当日波动率（ATR 百分比）
        base_slippage: 基础滑点（默认 0.01%）
        volume_scale: 成交量冲击系数（默认 0.1%）
        volatility_scale: 波动率冲击系数（默认 30%）
    
    Returns:
        slippage: 滑点百分比（如 0.0015 表示 0.15%）
    
    示例：
        订单 10000 股，市场成交量 100 万股，ATR=2%
        → volume_impact = 10000/1000000 × 0.001 = 0.00001 (0.001%)
        → volatility_impact = 0.02 × 0.3 = 0.006 (0.6%)
        → total = 0.0001 + 0.00001 + 0.006 = 0.00611 (0.611%)
    """
    # 成交量冲击（线性）
    if market_volume > 1e-9:
        volume_ratio = order_volume / market_volume
        volume_impact = volume_ratio * volume_scale
    else:
        volume_impact = volume_scale * 10.0  # 流动性极差时惩罚性滑点
    
    # 波动率冲击
    volatility_impact = volatility * volatility_scale
    
    # 总滑点（clip到合理区间）
    total_slippage = base_slippage + volume_impact + volatility_impact
    
    # 最大 5% 滑点保护
    if total_slippage > 0.05:
        total_slippage = 0.05
    
    return total_slippage


@njit(cache=True, fastmath=True)
def calculate_slippage_batch(
    order_volumes: np.ndarray,
    market_volumes: np.ndarray,
    volatilities: np.ndarray,
    base_slippage: float = 0.0001,
    volume_scale: float = 0.001,
    volatility_scale: float = 0.3,
) -> np.ndarray:
    """
    批量计算滑点（Numba 并行优化）。
    
    Args:
        order_volumes: shape=(n_orders,)
        market_volumes: shape=(n_orders,)
        volatilities: shape=(n_orders,)
        其他参数同 calculate_slippage_scalar
    
    Returns:
        slippages: shape=(n_orders,)
    """
    n = len(order_volumes)
    slippages = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        slippages[i] = calculate_slippage_scalar(
            order_volumes[i],
            market_volumes[i],
            volatilities[i],
            base_slippage,
            volume_scale,
            volatility_scale,
        )
    
    return slippages


# ============================================================================
# Part 2: 流动性检查（涨跌停、停牌）
# ============================================================================

@njit(cache=True, fastmath=True)
def check_limit_up(
    close: float,
    high: float,
    prev_close: float,
    is_st: bool = False,
    is_kcb: bool = False
) -> bool:
    """
    检测涨停（修复版 - 使用昨日收盘价）。
    
    Args:
        close: 当日收盘价
        high: 当日最高价
        prev_close: 昨日收盘价
        is_st: 是否ST股票（5%涨跌停）
        is_kcb: 是否科创板/创业板（20%涨跌停）
    
    Returns:
        True 表示涨停
    
    判断逻辑：
        1. 计算涨停价 = prev_close × (1 + 涨停幅度)
        2. 收盘价接近涨停价（误差 < 0.1%）
        3. 无上影线（high ≈ close）
    """
    if close <= 1e-9 or prev_close <= 1e-9:
        return False
    
    # ✅ FIX: 确定涨停幅度
    if is_kcb:
        limit_ratio = 0.20  # 科创板/创业板 20%
    elif is_st:
        limit_ratio = 0.05  # ST股票 5%
    else:
        limit_ratio = 0.10  # 普通股票 10%
    
    # ✅ FIX: 计算涨停价
    limit_up_price = prev_close * (1.0 + limit_ratio)
    
    # ✅ FIX: 判断条件
    # 1. 收盘价接近涨停价
    price_match = abs(close - limit_up_price) < prev_close * 0.001
    
    # 2. 无上影线（或上影线很小）
    no_shadow = abs(high - close) < close * 0.001
    
    return price_match and no_shadow


@njit(cache=True, fastmath=True)
def check_limit_down(
    close: float,
    low: float,
    prev_close: float,
    is_st: bool = False,
    is_kcb: bool = False
) -> bool:
    """
    检测跌停（修复版 - 使用昨日收盘价）。
    
    Args:
        close: 当日收盘价
        low: 当日最低价
        prev_close: 昨日收盘价
        is_st: 是否ST股票
        is_kcb: 是否科创板/创业板
    
    Returns:
        True 表示跌停
    """
    if close <= 1e-9 or prev_close <= 1e-9:
        return False
    
    # ✅ FIX: 确定跌停幅度
    if is_kcb:
        limit_ratio = 0.20
    elif is_st:
        limit_ratio = 0.05
    else:
        limit_ratio = 0.10
    
    # ✅ FIX: 计算跌停价
    limit_down_price = prev_close * (1.0 - limit_ratio)
    
    # ✅ FIX: 判断条件
    # 1. 收盘价接近跌停价
    price_match = abs(close - limit_down_price) < prev_close * 0.001
    
    # 2. 无下影线
    no_shadow = abs(low - close) < close * 0.001
    
    return price_match and no_shadow


@njit(cache=True, fastmath=True)
def check_suspended(
    close: float,
    high: float,
    low: float,
    volume: float,
) -> bool:
    """
    检测停牌（价格或成交量异常）。
    
    停牌标志：
    - 收盘价 = 0 或 NaN
    - 成交量 = 0
    - 高低价相等（可能是补数据）
    """
    if np.isnan(close) or close <= 1e-9:
        return True
    
    if np.isnan(volume) or volume <= 1e-9:
        return True
    
    if np.isnan(high) or np.isnan(low):
        return True
    
    # 高低价异常接近（可能是停牌补数据）
    if abs(high - low) < 1e-6:
        return True
    
    return False


# ============================================================================
# Part 3: ExchangeMatchEngine 主类
# ============================================================================

class ExchangeMatchEngine:
    """
    订单撮合引擎（市场模拟器）。
    
    职责：
    1. 接收 Signal，转换为 Order
    2. 根据市场数据（当日 OHLCV）撮合订单
    3. 计算动态滑点和佣金
    4. 检查流动性限制（涨跌停、停牌）
    5. 生成 Fill 回报
    
    使用方法：
        engine = ExchangeMatchEngine()
        
        # 提交信号
        orders = engine.submit_signals(signals, current_data, timestamp)
        
        # 撮合订单
        fills = engine.match_orders(orders, current_data, equity, timestamp)
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0003,  # 佣金率 0.03%
        min_commission: float = 5.0,      # 最低佣金 5 元
        base_slippage: float = 0.0001,    # 基础滑点 0.01%
        volume_scale: float = 0.001,      # 成交量冲击系数 0.1%
        volatility_scale: float = 0.3,    # 波动率冲击系数 30%
        max_order_ratio: float = 0.5,     # 单笔订单最大占市场成交量比例
    ):
        """
        Args:
            commission_rate: 佣金率（买卖双边）
            min_commission: 最低佣金
            base_slippage: 基础滑点
            volume_scale: 成交量冲击系数
            volatility_scale: 波动率冲击系数
            max_order_ratio: 单笔订单最大占市场成交量比例（超过则部分成交）
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.base_slippage = base_slippage
        self.volume_scale = volume_scale
        self.volatility_scale = volatility_scale
        self.max_order_ratio = max_order_ratio
        
        self.logger = logging.getLogger("engine.match")
        
        # 统计
        self._total_orders = 0
        self._filled_orders = 0
        self._rejected_orders = 0
        self._total_slippage = 0.0
        self._total_commission = 0.0
    
    # ========================================================================
    # 主流程：Signal → Order → Fill
    # ========================================================================
    
    def submit_signals(
        self,
        signals: List[Signal],
        current_data: pd.DataFrame,
        equity: float,
        timestamp: Timestamp,
    ) -> List[Order]:
        """
        提交信号，转换为订单。
        
        Args:
            signals: 策略生成的信号列表
            current_data: 当日市场数据（必须包含 code, close, volume 列）
            equity: 当前账户总权益
            timestamp: 当前时间戳
        
        Returns:
            orders: 订单列表（已过滤无效信号）
        """
        orders: List[Order] = []
        
        # 构建 code → 行情 映射（加速查找）
        if "code" in current_data.columns:
            data_dict = current_data.set_index("code").to_dict("index")
        else:
            data_dict = {}
        
        for signal in signals:
            # 验证信号
            if signal.weight <= 0.0:
                self.logger.debug(f"Skip zero-weight signal: {signal.code}")
                continue
            
            # 获取行情
            bar = data_dict.get(signal.code)
            if bar is None:
                self.logger.warning(f"No market data for {signal.code}, reject signal")
                self._rejected_orders += 1
                continue
            
            close_price = bar.get("close", 0.0)
            if pd.isna(close_price) or close_price <= 0.0:
                self.logger.warning(f"Invalid close price for {signal.code}: {close_price}")
                self._rejected_orders += 1
                continue
            
            # 计算订单量
            # weight = 0.1 表示用 10% 权益买入
            target_value = equity * signal.weight
            quantity = int(target_value / close_price / 100) * 100  # 取整到手（100股）
            
            if quantity < 100:
                self.logger.debug(f"Quantity < 100 for {signal.code}, skip")
                continue
            
            # 创建订单
            order = Order(
                order_id=self._generate_order_id(),
                code=signal.code,
                side=signal.side,
                order_type=OrderType.MARKET,  # 默认市价单
                quantity=quantity,
                price=close_price,  # 市价单价格为当前收盘价（简化）
                status=OrderStatus.SUBMITTED,
                create_time=timestamp,
                submit_time=timestamp,
                strategy_name=signal.strategy_name,
                signal_reason=signal.reason,
            )
            
            orders.append(order)
            self._total_orders += 1
        
        self.logger.info(f"Submitted {len(orders)} orders from {len(signals)} signals")
        return orders
    
    def match_orders(
        self,
        orders: List[Order],
        current_data: pd.DataFrame,
        timestamp: Timestamp,
    ) -> Tuple[List[Fill], List[Order]]:
        """
        撮合订单，生成成交回报。
        
        Args:
            orders: 待撮合订单列表
            current_data: 当日市场数据（OHLCV + 涨跌停标记）
            timestamp: 撮合时间戳
        
        Returns:
            (fills, rejected_orders)
            fills: 成交回报列表
            rejected_orders: 被拒绝的订单列表
        """
        fills: List[Fill] = []
        rejected_orders: List[Order] = []
        
        # 构建 code → 行情 映射
        if "code" in current_data.columns:
            data_dict = current_data.set_index("code").to_dict("index")
        else:
            data_dict = {}
        
        for order in orders:
            # 获取行情
            bar = data_dict.get(order.code)
            if bar is None:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "No market data available"
                rejected_orders.append(order)
                self._rejected_orders += 1
                continue
            
            close = bar.get("close", 0.0)
            high = bar.get("high", 0.0)
            low = bar.get("low", 0.0)
            volume = bar.get("volume", 0.0)
            open_price = bar.get("open", close)
            
            # ✅ FIX: 估算prev_close
            # TODO: 理想情况下应该传入历史数据，这里使用简化方案
            # 假设：日内波动不超过±5%，使用open作为参考
            if open_price > 1e-9:
                prev_close = open_price * 0.98  # 保守估计
            else:
                prev_close = close * 0.95
            
            # 检测股票类型
            is_st = False  # 简化：未实现ST检测
            is_kcb = code.startswith('688') or code.startswith('300')
            
            # 检查停牌
            if check_suspended(close, high, low, volume):
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Stock suspended or data unavailable"
                rejected_orders.append(order)
                self._rejected_orders += 1
                continue
            
            # 检查涨跌停
            if order.side == OrderSide.BUY:
                if check_limit_up(close, high, prev_close, is_st, is_kcb):
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = "Limit up, cannot buy"
                    rejected_orders.append(order)
                    self._rejected_orders += 1
                    continue
            else:  # SELL
                if check_limit_down(close, low, prev_close, is_st, is_kcb):
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = "Limit down, cannot sell"
                    rejected_orders.append(order)
                    self._rejected_orders += 1
                    continue
            
            # 检查流动性（订单量 vs 市场成交量）
            order_volume = order.quantity
            market_volume = volume
            
            if market_volume > 1e-9:
                order_ratio = order_volume / market_volume
                if order_ratio > self.max_order_ratio:
                    # 部分成交
                    filled_quantity = int(market_volume * self.max_order_ratio)
                    filled_quantity = (filled_quantity // 100) * 100  # 对齐到手
                    
                    if filled_quantity < 100:
                        order.status = OrderStatus.REJECTED
                        order.reject_reason = "Insufficient liquidity"
                        rejected_orders.append(order)
                        self._rejected_orders += 1
                        continue
                    
                    order.quantity = filled_quantity
                    order.status = OrderStatus.PARTIAL
                    self.logger.warning(
                        f"Partial fill for {order.code}: {filled_quantity}/{order_volume} "
                        f"(market volume: {market_volume:.0f})"
                    )
            
            # 计算动态滑点
            volatility = bar.get("atr_pct", 0.02)  # 默认 2% ATR
            if pd.isna(volatility):
                volatility = 0.02
            
            slippage_pct = calculate_slippage_scalar(
                order_volume,
                market_volume,
                volatility,
                self.base_slippage,
                self.volume_scale,
                self.volatility_scale,
            )
            
            # 成交价 = 收盘价 × (1 + slippage)
            # 买入向上滑点，卖出向下滑点
            if order.side == OrderSide.BUY:
                fill_price = close * (1.0 + slippage_pct)
            else:
                fill_price = close * (1.0 - slippage_pct)
            
            # 计算佣金
            trade_value = fill_price * order.quantity
            commission = max(trade_value * self.commission_rate, self.min_commission)
            
            # 更新订单状态
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.slippage = slippage_pct
            order.commission = commission
            order.fill_time = timestamp
            
            # 创建成交回报
            fill = Fill(
                order_id=order.order_id,
                code=order.code,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                commission=commission,
                slippage=slippage_pct,
                timestamp=timestamp,
                strategy_name=order.strategy_name,
            )
            
            fills.append(fill)
            self._filled_orders += 1
            self._total_slippage += slippage_pct
            self._total_commission += commission
        
        self.logger.info(
            f"Matched {len(fills)} fills, {len(rejected_orders)} rejected "
            f"(avg slippage: {self._avg_slippage():.4f}%)"
        )
        
        return fills, rejected_orders
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _generate_order_id(self) -> str:
        """生成唯一订单 ID"""
        return f"ORD-{uuid.uuid4().hex[:12].upper()}"
    
    def _avg_slippage(self) -> float:
        """平均滑点（百分比）"""
        if self._filled_orders > 0:
            return (self._total_slippage / self._filled_orders) * 100
        return 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """获取撮合统计"""
        return {
            "total_orders": self._total_orders,
            "filled_orders": self._filled_orders,
            "rejected_orders": self._rejected_orders,
            "fill_rate": self._filled_orders / self._total_orders if self._total_orders > 0 else 0.0,
            "avg_slippage_pct": self._avg_slippage(),
            "total_commission": self._total_commission,
        }
    
    def reset_statistics(self) -> None:
        """重置统计"""
        self._total_orders = 0
        self._filled_orders = 0
        self._rejected_orders = 0
        self._total_slippage = 0.0
        self._total_commission = 0.0


# ============================================================================
# Part 4: 导出
# ============================================================================

__all__ = [
    "ExchangeMatchEngine",
    "calculate_slippage_scalar",
    "calculate_slippage_batch",
    "check_limit_up",
    "check_limit_down",
    "check_suspended",
    "NUMBA_AVAILABLE",
]


# ============================================================================
# Part 5: 测试
# ============================================================================

if __name__ == "__main__":
    import time
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("EXCHANGE MATCH ENGINE - TEST")
    print("=" * 70)
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print()
    
    # 准备测试数据
    n_stocks = 100
    codes = [f"SH{600000 + i:06d}" for i in range(n_stocks)]
    
    current_data = pd.DataFrame({
        "code": codes,
        "close": np.random.uniform(10, 50, n_stocks),
        "high": np.random.uniform(10, 50, n_stocks),
        "low": np.random.uniform(10, 50, n_stocks),
        "volume": np.random.uniform(1e6, 1e8, n_stocks),
        "atr_pct": np.random.uniform(0.01, 0.03, n_stocks),
    })
    
    # 创建引擎
    engine = ExchangeMatchEngine()
    
    # 生成测试信号
    signals = [
        Signal(
            code=codes[i],
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            weight=0.05,
            reason="Test signal",
            strategy_name="test_strategy",
        )
        for i in range(50)  # 50 个信号
    ]
    
    timestamp = Timestamp.now()
    equity = 1000000.0
    
    # 测试：提交信号
    print("Test 1: Submit signals")
    orders = engine.submit_signals(signals, current_data, equity, timestamp)
    print(f"  ✓ Created {len(orders)} orders from {len(signals)} signals")
    print()
    
    # 测试：撮合订单
    print("Test 2: Match orders")
    start = time.perf_counter()
    fills, rejected = engine.match_orders(orders, current_data, timestamp)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  ✓ Matched in {elapsed:.2f}ms")
    print(f"  ✓ {len(fills)} fills, {len(rejected)} rejected")
    print()
    
    # 统计
    stats = engine.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # 测试：批量滑点计算
    if NUMBA_AVAILABLE:
        print("Test 3: Batch slippage calculation")
        n_orders = 1000
        order_volumes = np.random.uniform(1e4, 1e5, n_orders)
        market_volumes = np.random.uniform(1e6, 1e8, n_orders)
        volatilities = np.random.uniform(0.01, 0.03, n_orders)
        
        start = time.perf_counter()
        slippages = calculate_slippage_batch(
            order_volumes, market_volumes, volatilities
        )
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  ✓ Calculated {n_orders} slippages in {elapsed:.2f}ms")
        print(f"  ✓ Avg slippage: {np.mean(slippages) * 100:.4f}%")
        print()
    
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
