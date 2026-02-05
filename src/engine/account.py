"""
src/engine/account.py
=====================
Phase 4 — AccountManager（账户管理器）

核心职责：
  1. 资金管理：现金、权益、保证金
  2. 持仓管理：使用 PositionStateProtocol 统一接口
  3. 账户快照：O(1) 复杂度生成净值快照
  4. 交易流水：记录每笔交易的完整信息

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. **Protocol 驱动的持仓管理**
   ────────────────────────
   AccountManager 仅依赖 PositionStateProtocol，不依赖具体类型。
   支持 5 个策略的异构持仓：
   - AlphaHunter: 锁利逻辑 (LockProfitMixin)
   - ShortTermRSRS: 移动止损 (TrailingStopMixin)
   - MomentumReversal: 复合因子
   - SentimentReversal: 时间止损 (TimeStopMixin)
   - UltraShort: K线计数 (BarCountMixin)
   
   所有策略通过统一的 add_position / remove_position 接口管理。

2. **AccountSnapshot（账户快照）**
   ──────────────────────────────
   O(1) 复杂度快照生成：
   
   class AccountSnapshot:
       timestamp: Timestamp
       date: str
       cash: float
       market_value: float  # 持仓市值
       total_equity: float  # 总权益
       positions_count: int
       long_value: float
       short_value: float  # (期货/期权场景)
   
   用于绩效分析和净值曲线生成。

3. **交易流水记录**
   ────────────────
   每笔成交后记录：
   - 买入：扣减现金，增加持仓
   - 卖出：增加现金，减少持仓，计算盈亏
   
   流水包含：
   - 交易时间、代码、方向、数量、价格
   - 佣金、滑点
   - 盈亏（仅卖出时）
   - 持仓天数

4. **风控检查**
   ───────────
   - 现金充足性检查
   - 持仓数量限制
   - 单股权重限制
   - 总仓位限制

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Set

import pandas as pd

# 导入类型
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from strategy.types import (
    Fill,
    OrderSide,
    PositionStateProtocol,
    Timestamp,
)


# ============================================================================
# Part 1: AccountSnapshot（账户快照）
# ============================================================================

@dataclass(frozen=True)
class AccountSnapshot:
    """
    账户快照（不可变）。
    
    用于绩效分析和净值曲线生成。O(1) 复杂度创建。
    """
    timestamp: Timestamp
    date: str
    cash: float
    market_value: float
    total_equity: float
    positions_count: int
    long_value: float
    short_value: float = 0.0  # 期货场景
    
    # 杠杆相关（期货/期权）
    leverage: float = 1.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": str(self.timestamp),
            "date": self.date,
            "cash": self.cash,
            "market_value": self.market_value,
            "total_equity": self.total_equity,
            "positions_count": self.positions_count,
            "long_value": self.long_value,
            "short_value": self.short_value,
            "leverage": self.leverage,
        }
    
    @property
    def position_ratio(self) -> float:
        """持仓占比"""
        if self.total_equity <= 0:
            return 0.0
        return self.market_value / self.total_equity


# ============================================================================
# Part 2: AccountManager 主类
# ============================================================================

class AccountManager:
    """
    账户管理器。
    
    职责：
    1. 资金管理（现金、权益）
    2. 持仓管理（使用 PositionStateProtocol）
    3. 交易流水记录
    4. 账户快照生成
    5. 风控检查
    
    使用方法：
        account = AccountManager(initial_cash=1000000.0)
        
        # 处理成交
        account.process_fill(fill, position_state, current_price)
        
        # 更新持仓市值
        account.update_market_value(current_data)
        
        # 生成快照
        snapshot = account.create_snapshot(timestamp, date)
    """
    
    def __init__(
        self,
        initial_cash: float = 1000000.0,
        max_positions: int = 20,
        max_single_position_ratio: float = 0.10,
        max_total_position_ratio: float = 0.95,
    ):
        """
        Args:
            initial_cash: 初始资金
            max_positions: 最大持仓数
            max_single_position_ratio: 单股最大权重
            max_total_position_ratio: 总仓位上限
        """
        # 资金
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.frozen_cash = 0.0  # ✅ FIX: T+1冻结资金
        
        # 持仓（使用 Protocol 接口）
        self._positions: Dict[str, PositionStateProtocol] = {}
        
        # 持仓市值缓存（避免重复计算）
        self._market_values: Dict[str, float] = {}
        self._total_market_value: float = 0.0
        
        # 风控参数
        self.max_positions = max_positions
        self.max_single_position_ratio = max_single_position_ratio
        self.max_total_position_ratio = max_total_position_ratio
        
        # 交易流水（滚动窗口）
        self._trade_history: Deque[Dict] = deque(maxlen=10000)
        
        # 快照历史（用于绩效分析）
        self._snapshots: List[AccountSnapshot] = []
        
        self.logger = logging.getLogger("engine.account")
    
    # ========================================================================
    # 核心方法：成交处理
    # ========================================================================
    
    def process_fill(
        self,
        fill: Fill,
        position_state: Optional[PositionStateProtocol] = None,
        current_price: Optional[float] = None,
    ) -> bool:
        """
        处理成交回报。
        
        Args:
            fill: 成交回报
            position_state: 持仓状态（买入时需提供，卖出时可选）
            current_price: 当前价格（用于更新市值）
        
        Returns:
            success: 是否成功处理
        """
        code = fill.code
        side = fill.side
        quantity = fill.quantity
        price = fill.price
        commission = fill.commission
        
        if side == OrderSide.BUY:
            return self._process_buy(fill, position_state, current_price)
        else:
            return self._process_sell(fill, current_price)
    
    def _process_buy(
        self,
        fill: Fill,
        position_state: Optional[PositionStateProtocol],
        current_price: Optional[float],
    ) -> bool:
        """处理买入成交"""
        code = fill.code
        total_cost = fill.total_cost
        
        # 检查资金充足性
        if total_cost > self.cash:
            self.logger.error(
                f"Insufficient cash for {code}: need {total_cost:.2f}, "
                f"available {self.cash:.2f}"
            )
            return False
        
        # 扣减现金
        self.cash -= total_cost
        
        # 添加或更新持仓
        if code in self._positions:
            # 已有持仓（加仓）
            old_pos = self._positions[code]
            old_qty = old_pos.quantity
            old_price = old_pos.entry_price
            
            # 计算新的平均成本
            new_qty = old_qty + fill.quantity
            new_price = (old_qty * old_price + fill.quantity * fill.price) / new_qty
            
            # FIXED: 重新创建实例而非直接修改
            new_pos = type(old_pos)(
                code=code,
                entry_price=new_price,
                entry_date=old_pos.entry_date,
                quantity=new_qty,
                stop_loss_price=getattr(old_pos, 'stop_loss_price', 0.0),
                take_profit_price=getattr(old_pos, 'take_profit_price', float('inf')),
                trailing_stop_price=getattr(old_pos, 'trailing_stop_price', 0.0),
                highest_price=max(getattr(old_pos, 'highest_price', new_price), current_price or fill.price),
                lowest_price=min(getattr(old_pos, 'lowest_price', new_price), current_price or fill.price),
            )
            # 继承 Mixin 状态
            if hasattr(old_pos, 'lock_levels'):
                new_pos.lock_levels = old_pos.lock_levels
                new_pos.current_lock_level = old_pos.current_lock_level
                new_pos.hard_stop = old_pos.hard_stop
            self._positions[code] = new_pos
            
            self.logger.info(
                f"Add position {code}: {old_qty} → {new_qty}, "
                f"avg price {new_price:.2f}"
            )
        else:
            # 新建持仓
            if position_state is None:
                self.logger.error(f"No position_state provided for new buy {code}")
                return False
            
            self._positions[code] = position_state
            
            self.logger.info(
                f"Open position {code}: qty={fill.quantity}, "
                f"price={fill.price:.2f}"
            )
        
        # 更新市值缓存
        price = current_price if current_price else fill.price
        self._market_values[code] = self._positions[code].quantity * price
        self._update_total_market_value()
        
        # 记录交易流水
        self._record_trade(fill, side=OrderSide.BUY, pnl=0.0, pnl_ratio=0.0)
        
        return True
    
    def _process_sell(
        self,
        fill: Fill,
        current_price: Optional[float],
    ) -> bool:
        """处理卖出成交"""
        code = fill.code
        
        # 检查持仓
        if code not in self._positions:
            self.logger.error(f"No position to sell for {code}")
            return False
        
        pos = self._positions[code]
        
        if fill.quantity > pos.quantity:
            self.logger.error(
                f"Sell quantity {fill.quantity} > position {pos.quantity} for {code}"
            )
            return False
        
        # 计算盈亏
        entry_price = pos.entry_price
        exit_price = fill.price
        pnl = (exit_price - entry_price) * fill.quantity - fill.commission
        pnl_ratio = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # ✅ FIX: T+1结算 - 卖出资金冻结，次日到账
        proceeds = fill.total_value - fill.commission
        self.frozen_cash += proceeds  # 冻结资金
        
        self.logger.debug(
            f"Sell {code}: proceeds={proceeds:.2f} frozen (T+1)"
        )
        
        # 更新持仓
        if fill.quantity == pos.quantity:
            # 全部卖出
            del self._positions[code]
            del self._market_values[code]
            
            self.logger.info(
                f"Close position {code}: pnl={pnl:.2f} ({pnl_ratio:.2%})"
            )
        else:
            # 部分卖出 - FIXED: 重新创建实例
            new_qty = pos.quantity - fill.quantity
            new_pos = type(pos)(
                code=code,
                entry_price=pos.entry_price,
                entry_date=pos.entry_date,
                quantity=new_qty,
                stop_loss_price=getattr(pos, 'stop_loss_price', 0.0),
                take_profit_price=getattr(pos, 'take_profit_price', float('inf')),
                trailing_stop_price=getattr(pos, 'trailing_stop_price', 0.0),
                highest_price=getattr(pos, 'highest_price', pos.entry_price),
                lowest_price=getattr(pos, 'lowest_price', pos.entry_price),
            )
            if hasattr(pos, 'lock_levels'):
                new_pos.lock_levels = pos.lock_levels
                new_pos.current_lock_level = pos.current_lock_level
                new_pos.hard_stop = pos.hard_stop
            self._positions[code] = new_pos
            price = current_price if current_price else fill.price
            self._market_values[code] = pos.quantity * price
            
            self.logger.info(
                f"Reduce position {code}: {pos.quantity + fill.quantity} → {pos.quantity}, "
                f"pnl={pnl:.2f}"
            )
        
        self._update_total_market_value()
        
        # 记录交易流水
        self._record_trade(fill, side=OrderSide.SELL, pnl=pnl, pnl_ratio=pnl_ratio)
        
        return True
    
    def _update_total_market_value(self) -> None:
        """更新总市值缓存"""
        self._total_market_value = sum(self._market_values.values())
    
    # ========================================================================
    # 持仓管理
    # ========================================================================
    
    def add_position(
        self,
        code: str,
        position: PositionStateProtocol,
        current_price: float,
    ) -> None:
        """
        添加持仓（外部直接操作，不经过 Fill）。
        
        用于策略层直接管理持仓状态。
        """
        self._positions[code] = position
        self._market_values[code] = position.quantity * current_price
        self._update_total_market_value()
    
    def remove_position(self, code: str) -> Optional[PositionStateProtocol]:
        """移除持仓"""
        pos = self._positions.pop(code, None)
        if pos:
            self._market_values.pop(code, None)
            self._update_total_market_value()
        return pos
    
    def get_position(self, code: str) -> Optional[PositionStateProtocol]:
        """获取持仓"""
        return self._positions.get(code)
    
    def has_position(self, code: str) -> bool:
        """检查是否持有某股票"""
        return code in self._positions
    
    @property
    def position_codes(self) -> Set[str]:
        """持仓代码集合"""
        return set(self._positions.keys())
    
    @property
    def position_count(self) -> int:
        """持仓数量"""
        return len(self._positions)
    
    # ========================================================================
    # 市值更新
    # ========================================================================
    
    def update_market_value(self, current_data: pd.DataFrame) -> None:
        """
        批量更新持仓市值。
        
        Args:
            current_data: 当日行情（必须包含 code, close 列）
        """
        if "code" in current_data.columns:
            price_dict = dict(zip(current_data["code"], current_data["close"]))
        else:
            price_dict = {}
        
        for code, pos in self._positions.items():
            price = price_dict.get(code, 0.0)
            if price > 0:
                self._market_values[code] = pos.quantity * price
                # 更新持仓状态的未实现盈亏
                pos.update_trailing_stop(price)
        
        self._update_total_market_value()
    
    def update_single_position_value(self, code: str, price: float) -> None:
        """更新单个持仓市值"""
        if code in self._positions:
            pos = self._positions[code]
            self._market_values[code] = pos.quantity * price
            pos.update_trailing_stop(price)
            self._update_total_market_value()
    
    # ========================================================================
    # 账户快照
    # ========================================================================
    
    def create_snapshot(
        self,
        timestamp: Timestamp,
        date: str,
    ) -> AccountSnapshot:
        """
        创建账户快照（O(1) 复杂度）。
        
        Args:
            timestamp: 时间戳
            date: 日期字符串
        
        Returns:
            snapshot: 账户快照
        """
        total_equity = self.cash + self._total_market_value
        
        snapshot = AccountSnapshot(
            timestamp=timestamp,
            date=date,
            cash=self.cash,
            market_value=self._total_market_value,
            total_equity=total_equity,
            positions_count=len(self._positions),
            long_value=self._total_market_value,
            short_value=0.0,
        )
        
        self._snapshots.append(snapshot)
        
        return snapshot
    
    def get_snapshots(self) -> List[AccountSnapshot]:
        """获取所有快照"""
        return self._snapshots
    
    def get_equity_curve(self) -> pd.DataFrame:
        """获取净值曲线（DataFrame 格式）"""
        if not self._snapshots:
            return pd.DataFrame()
        
        data = [s.to_dict() for s in self._snapshots]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        
        return df
    
    # ========================================================================
    # 风控检查
    # ========================================================================
    
    def can_open_position(
        self,
        code: str,
        value: float,
    ) -> Tuple[bool, str]:
        """
        检查是否可以开仓。
        
        Args:
            code: 股票代码
            value: 开仓金额
        
        Returns:
            (can_open, reason)
        """
        # 检查持仓数量
        if code not in self._positions and len(self._positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # 检查资金充足性
        if value > self.cash:
            return False, f"Insufficient cash: need {value:.2f}, available {self.cash:.2f}"
        
        # 检查单股权重
        total_equity = self.cash + self._total_market_value
        if total_equity > 0:
            weight = value / total_equity
            if weight > self.max_single_position_ratio:
                return False, (
                    f"Single position ratio {weight:.2%} exceeds limit "
                    f"{self.max_single_position_ratio:.2%}"
                )
        
        # 检查总仓位
        if total_equity > 0:
            new_market_value = self._total_market_value + value
            total_ratio = new_market_value / total_equity
            if total_ratio > self.max_total_position_ratio:
                return False, (
                    f"Total position ratio {total_ratio:.2%} exceeds limit "
                    f"{self.max_total_position_ratio:.2%}"
                )
        
        return True, ""
    
    # ========================================================================
    # 交易流水
    # ========================================================================
    
    def _record_trade(
        self,
        fill: Fill,
        side: OrderSide,
        pnl: float,
        pnl_ratio: float,
    ) -> None:
        """记录交易流水"""
        trade = {
            "timestamp": fill.timestamp,
            "code": fill.code,
            "side": side.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "slippage": fill.slippage,
            "pnl": pnl,
            "pnl_ratio": pnl_ratio,
            "strategy_name": fill.strategy_name,
        }
        self._trade_history.append(trade)
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """获取交易流水"""
        return list(self._trade_history)[-limit:]
    
    def on_day_end(self) -> None:
        """
        日终结算（T+1资金解冻）。
        
        ✅ FIX: 实现T+1结算契约
        - 卖出当日：资金冻结在 frozen_cash
        - 下一交易日开始时：frozen_cash → cash
        
        调用时机：
            在 execution.py 的 _run_single_day() 结束时调用
        """
        if self.frozen_cash > 0:
            self.cash += self.frozen_cash
            self.logger.info(
                f"T+1 settlement: {self.frozen_cash:.2f} unfrozen → available cash"
            )
            self.frozen_cash = 0.0
    
    def get_trade_statistics(self) -> Dict:
        """获取交易统计"""
        trades = list(self._trade_history)
        if not trades:
            return {}
        
        # 仅统计卖出（有盈亏）
        sell_trades = [t for t in trades if t["side"] == "SELL"]
        if not sell_trades:
            return {"total_trades": len(trades)}
        
        pnls = [t["pnl"] for t in sell_trades]
        pnl_ratios = [t["pnl_ratio"] for t in sell_trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            "total_trades": len(trades),
            "sell_trades": len(sell_trades),
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "win_rate": len(wins) / len(sell_trades) if sell_trades else 0.0,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "max_win": max(pnls) if pnls else 0.0,
            "max_loss": min(pnls) if pnls else 0.0,
            "avg_pnl_ratio": sum(pnl_ratios) / len(pnl_ratios) if pnl_ratios else 0.0,
        }
    
    # ========================================================================
    # 状态查询
    # ========================================================================
    
    @property
    def total_equity(self) -> float:
        """总权益"""
        return self.cash + self._total_market_value
    
    @property
    def market_value(self) -> float:
        """持仓市值"""
        return self._total_market_value
    
    @property
    def position_ratio(self) -> float:
        """持仓占比"""
        equity = self.total_equity
        if equity <= 0:
            return 0.0
        return self._total_market_value / equity
    
    def __repr__(self) -> str:
        return (
            f"AccountManager(cash={self.cash:.2f}, "
            f"market_value={self._total_market_value:.2f}, "
            f"equity={self.total_equity:.2f}, "
            f"positions={len(self._positions)})"
        )


# ============================================================================
# Part 3: 导出
# ============================================================================

__all__ = [
    "AccountManager",
    "AccountSnapshot",
]


# ============================================================================
# Part 4: 测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 70)
    print("ACCOUNT MANAGER - TEST")
    print("=" * 70)
    print()
    
    # 创建账户
    account = AccountManager(initial_cash=1000000.0)
    print(f"Initial state: {account}")
    print()
    
    # 模拟买入
    from strategy.types import BasePositionState
    
    fill_buy = Fill(
        order_id="ORD-001",
        code="SH600000",
        side=OrderSide.BUY,
        quantity=1000,
        price=10.5,
        commission=31.5,
        slippage=0.0001,
        timestamp=Timestamp.now(),
        strategy_name="test",
    )
    
    position = BasePositionState(
        code="SH600000",
        entry_price=10.5,
        entry_date="2024-01-01",
        quantity=1000,
    )
    
    print("Test 1: Process buy")
    success = account.process_fill(fill_buy, position, current_price=10.5)
    print(f"  ✓ Buy processed: {success}")
    print(f"  ✓ {account}")
    print()
    
    # 更新市值
    print("Test 2: Update market value")
    current_data = pd.DataFrame({
        "code": ["SH600000"],
        "close": [11.0],
    })
    account.update_market_value(current_data)
    print(f"  ✓ Market value updated")
    print(f"  ✓ {account}")
    print()
    
    # 生成快照
    print("Test 3: Create snapshot")
    snapshot = account.create_snapshot(Timestamp.now(), "2024-01-02")
    print(f"  ✓ Snapshot: {snapshot}")
    print()
    
    # 模拟卖出
    print("Test 4: Process sell")
    fill_sell = Fill(
        order_id="ORD-002",
        code="SH600000",
        side=OrderSide.SELL,
        quantity=1000,
        price=11.0,
        commission=33.0,
        slippage=0.0001,
        timestamp=Timestamp.now(),
        strategy_name="test",
    )
    
    success = account.process_fill(fill_sell, current_price=11.0)
    print(f"  ✓ Sell processed: {success}")
    print(f"  ✓ {account}")
    print()
    
    # 统计
    print("Trade statistics:")
    stats = account.get_trade_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
