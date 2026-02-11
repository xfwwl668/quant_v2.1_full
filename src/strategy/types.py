"""
strategy/types.py
=================
Phase 1 — 基础层：所有核心类型定义的唯一来源。

设计原则：
  1. 热路径类（Signal, Order, Fill, PositionState 及其子类）全部使用 __slots__，
     消除 __dict__ 分配开销，对 GC 压力和 cache-line 友好性均有显著收益。
  2. Timestamp 为 frozen dataclass（不可变），线程安全可共享。
  3. FactorAccessor 是策略层与 Numba 因子引擎之间的唯一桥接层——
     见 FactorAccessor 类的详细注释。
  4. 异常体系扁平，5层继承封顶，便于 except 精确捕捉。

依赖：
  仅依赖 stdlib + numpy + pandas。不引入 numba。
  与 Numba 引擎的对接发生在 FactorAccessor 内部（ndarray → float 索引查找）。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd


# ============================================================================
# Part 1: 枚举类型
# ============================================================================


class OrderSide(Enum):
    """订单方向"""

    BUY = "BUY"
    SELL = "SELL"

    def __str__(self) -> str:
        return self.value

    @property
    def opposite(self) -> "OrderSide":
        """返回反向"""
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class OrderType(Enum):
    """订单类型"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """订单状态机"""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Timeframe(Enum):
    """时间周期枚举，覆盖 UltraShort(1m) → AlphaHunter(1d) 全部层级"""

    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1w"

    @property
    def minutes(self) -> int:
        """转换为分钟数，用于 TimeAligner 多周期对齐"""
        _MAP: Dict[str, int] = {
            "tick": 0,
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "1d": 1440,
            "1w": 10080,
        }
        return _MAP.get(self.value, 1440)

    @property
    def is_intraday(self) -> bool:
        """日内级别标记，HybridExecutionEngine 据此路由到事件驱动通道"""
        return self.minutes < 1440


class StrategyState(Enum):
    """策略生命周期状态机"""

    CREATED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()


# ============================================================================
# Part 2: Timestamp — 不可变、毫秒精度统一时间戳
# ============================================================================


@dataclass(frozen=True)
class Timestamp:
    """
    统一时间戳（毫秒精度，不可变）。

    frozen=True 保证线程安全，可安全共享给 Numba parallel 子线程。
    Order / Fill 均持有此对象；回测引擎构建时一次创建，策略层只读。
    """

    __slots__ = ("value",)

    value: int  # Unix 毫秒时间戳

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def from_datetime(cls, dt: datetime) -> "Timestamp":
        return cls(int(dt.timestamp() * 1000))

    @classmethod
    def from_date(cls, d: date, hour: int = 15, minute: int = 0) -> "Timestamp":
        """默认取收盘时间 15:00"""
        dt = datetime.combine(d, dt_time(hour, minute))
        return cls.from_datetime(dt)

    @classmethod
    def from_str(cls, s: str, fmt: str = "%Y-%m-%d") -> "Timestamp":
        """从日期字符串创建，默认格式 YYYY-MM-DD，取 15:00"""
        d = datetime.strptime(s, fmt).date()
        return cls.from_date(d)

    @classmethod
    def now(cls) -> "Timestamp":
        return cls.from_datetime(datetime.now())

    # ------------------------------------------------------------------
    # 转换
    # ------------------------------------------------------------------
    def to_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.value / 1000.0)

    def to_date(self) -> date:
        return self.to_datetime().date()

    def to_date_str(self) -> str:
        return self.to_date().isoformat()

    # ------------------------------------------------------------------
    # 运算与比较
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return self.to_datetime().strftime("%Y-%m-%d %H:%M:%S")

    def __lt__(self, other: "Timestamp") -> bool:  # type: ignore[override]
        return self.value < other.value

    def __le__(self, other: "Timestamp") -> bool:  # type: ignore[override]
        return self.value <= other.value

    def __add__(self, milliseconds: int) -> "Timestamp":
        return Timestamp(self.value + milliseconds)

    def __sub__(self, other: Union["Timestamp", int]) -> Union[int, "Timestamp"]:
        if isinstance(other, Timestamp):
            return self.value - other.value  # 返回毫秒差
        return Timestamp(self.value - other)


# ============================================================================
# Part 3: Signal / Order / Fill — 交易流水线核心数据类
#
# 生命周期：Strategy.generate_signals() → Signal
#           ExchangeMatchEngine.submit()  → Order
#           ExchangeMatchEngine.match()   → Fill → on_order_filled 回调
#
# 所有三者均使用 __slots__。Signal 用 frozen 保证信号发出后不可修改。
# ============================================================================


class Signal:
    """
    交易信号（不可变快速对象）。

    frozen dataclass 会产生额外 __hash__ 开销；这里手写 __slots__ class，
    在 __init__ 里用 object.__setattr__ 模拟 frozen 语义，同时保持 __slots__ 效率。

    weight = 0.0 表示清仓信号；priority 越大越先被 MatchEngine 处理。
    """

    __slots__ = (
        "code",
        "side",
        "weight",
        "price",
        "reason",
        "priority",
        "timestamp",
        "strategy_name",
        "metadata",
    )

    def __init__(
        self,
        code: str,
        side: OrderSide,
        weight: float,
        price: Optional[float] = None,
        reason: str = "",
        priority: int = 0,
        timestamp: Optional[Timestamp] = None,
        strategy_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if weight < 0.0:
            raise ValueError(f"Signal weight must be >= 0, got {weight}")
        if weight > 1.0:
            raise ValueError(f"Signal weight must be <= 1, got {weight}")

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "strategy_name", strategy_name)
        object.__setattr__(self, "metadata", metadata if metadata is not None else {})

    # 禁止外部修改，模拟 frozen
    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("Signal is immutable after creation")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("Signal is immutable after creation")

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Signal({self.code}, {self.side.value}, "
            f"weight={self.weight:.2%}, reason='{self.reason[:40]}')"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return NotImplemented
        return (
            self.code == other.code
            and self.side == other.side
            and self.weight == other.weight
            and self.strategy_name == other.strategy_name
        )

    def __hash__(self) -> int:
        return hash((self.code, self.side, self.weight, self.strategy_name))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "side": self.side.value,
            "weight": self.weight,
            "price": self.price,
            "reason": self.reason,
            "priority": self.priority,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "strategy_name": self.strategy_name,
            "metadata": self.metadata,
        }


class Order:
    """
    订单对象。由 MatchEngine 从 Signal 转换创建，状态可变（PENDING→FILLED）。
    使用 __slots__ 避免 __dict__ 开销；MatchEngine 在紧密循环中大量创建此对象。
    """

    __slots__ = (
        "order_id",
        "code",
        "side",
        "order_type",
        "quantity",
        "price",
        "stop_price",
        # 状态
        "status",
        "filled_quantity",
        "filled_price",
        "commission",
        "slippage",
        # 时间
        "create_time",
        "create_date",
        "submit_time",
        "fill_time",
        # 来源追溯
        "strategy_name",
        "signal_reason",
        # 拒绝信息
        "reject_reason",
    )

    def __init__(
        self,
        order_id: str,
        code: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        status: OrderStatus = OrderStatus.PENDING,
        filled_quantity: int = 0,
        filled_price: float = 0.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        create_time: Optional[Timestamp] = None,
        create_date: str = "",
        submit_time: Optional[Timestamp] = None,
        fill_time: Optional[Timestamp] = None,
        strategy_name: str = "",
        signal_reason: str = "",
        reject_reason: str = "",
    ) -> None:
        self.order_id = order_id
        self.code = code
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.status = status
        self.filled_quantity = filled_quantity
        self.filled_price = filled_price
        self.commission = commission
        self.slippage = slippage
        self.create_time = create_time
        self.submit_time = submit_time
        self.fill_time = fill_time
        self.strategy_name = strategy_name
        self.signal_reason = signal_reason
        self.reject_reason = reject_reason
        # create_date 自动推导
        self.create_date = create_date if create_date else (
            create_time.to_date_str() if create_time else ""
        )

    # ------------------------------------------------------------------
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_terminal(self) -> bool:
        """终态：不再会有状态变更"""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )

    def __repr__(self) -> str:
        return (
            f"Order({self.order_id}, {self.code}, {self.side.value}, "
            f"qty={self.quantity}, status={self.status.value})"
        )


class Fill:
    """
    成交回报。由 MatchEngine 在确认成交时创建，传递给 on_order_filled 回调。
    __slots__ 保证极低的对象分配代价。
    """

    __slots__ = (
        "order_id",
        "code",
        "side",
        "quantity",
        "price",
        "commission",
        "slippage",
        "timestamp",
        "strategy_name",
    )

    def __init__(
        self,
        order_id: str,
        code: str,
        side: OrderSide,
        quantity: int,
        price: float,
        commission: float,
        slippage: float,
        timestamp: Timestamp,
        strategy_name: str = "",
    ) -> None:
        self.order_id = order_id
        self.code = code
        self.side = side
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.slippage = slippage
        self.timestamp = timestamp
        self.strategy_name = strategy_name

    # ------------------------------------------------------------------
    @property
    def total_value(self) -> float:
        """成交金额（不含手续费）"""
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        """总成本 = 成交金额 + 佣金"""
        return self.total_value + self.commission

    def __repr__(self) -> str:
        return (
            f"Fill({self.code}, {self.side.value}, "
            f"qty={self.quantity}, price={self.price:.2f}, "
            f"commission={self.commission:.4f})"
        )


# ============================================================================
# Part 4: PositionState 协议 + 基础实现 + Mixin 组合
#
# 核心设计思路（解决 5 策略散乱持仓问题的关键）：
#
#   PositionStateProtocol（Protocol）
#       └── 约束最小公约数字段：code / entry_price / entry_date / quantity
#           + update_trailing_stop 方法签名
#       引擎层（AccountManager, ExchangeMatchEngine）仅依赖此协议；
#       不 import 任何具体 PositionState 子类。
#
#   BasePositionState（__slots__ 类）
#       └── 实现协议的默认行为，包含 止损/止盈/高低价/未实现盈亏 等通用字段。
#
#   Mixin（轻量功能插件）
#       ├── TrailingStopMixin   → 移动止损逻辑（ATR 动态）
#       ├── LockProfitMixin     → 分级锁利逻辑（AlphaHunter 风格）
#       ├── TimeStopMixin       → 时间止损（SentimentReversal 风格）
#       └── BarCountMixin       → K线计数止损（UltraShort 风格）
#
#   策略级具体类（在 Phase 7 策略适配时定义）：
#       AlphaHunterPosition(BasePositionState, TrailingStopMixin, LockProfitMixin)
#       ShortTermPosition  (BasePositionState, TrailingStopMixin)
#       MomentumPosition   (BasePositionState, TrailingStopMixin)  # 原来用3个平行dict
#       SentimentPosition  (BasePositionState, TimeStopMixin)
#       UltraShortPosition (BasePositionState, BarCountMixin)
#
# __slots__ 策略：
#   Protocol 本身不能开 __slots__（typing 限制）。
#   BasePositionState + Mixin 均声明自己的 __slots__；
#   子类合并时 MRO 自动合并 slots，无 __dict__ 泄漏。
# ============================================================================


@runtime_checkable
class PositionStateProtocol(Protocol):
    """
    持仓状态协议——引擎层唯一依赖的持仓接口。

    runtime_checkable 使得 isinstance(pos, PositionStateProtocol) 可用于
    AccountManager 中的运行时校验，无需引入具体类型。
    """

    code: str
    entry_price: float
    entry_date: str
    quantity: int

    def update_trailing_stop(self, current_price: float, **kwargs: Any) -> None:
        """更新移动止损。子类根据自身策略逻辑实现。"""
        ...


class BasePositionState:
    """
    基础持仓状态——所有具体 PositionState 的继承根。

    包含通用字段和默认实现；子类通过 Mixin 组合扩展行为。
    不使用 dataclass，手写 __init__ + __slots__ 以获得最大化的创建吞吐。
    """

    __slots__ = (
        "code",
        "entry_price",
        "entry_date",
        "quantity",
        # 止损止盈价格
        "stop_loss_price",
        "take_profit_price",
        "trailing_stop_price",
        # 价格跟踪
        "highest_price",
        "lowest_price",
        # 盈亏
        "unrealized_pnl",
        "realized_pnl",
    )

    def __init__(
        self,
        code: str,
        entry_price: float,
        entry_date: str,
        quantity: int,
        stop_loss_price: float = 0.0,
        take_profit_price: float = math.inf,
        trailing_stop_price: float = 0.0,
        highest_price: float = 0.0,
        lowest_price: float = math.inf,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> None:
        self.code = code
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.quantity = quantity
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.trailing_stop_price = trailing_stop_price
        # 高低价初始化：未传入时取入场价
        self.highest_price = highest_price if highest_price != 0.0 else entry_price
        self.lowest_price = lowest_price if lowest_price != math.inf else entry_price
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl

    # ------------------------------------------------------------------
    # 协议方法实现
    # ------------------------------------------------------------------
    def update_trailing_stop(
        self,
        current_price: float,
        atr: float = 0.0,
        multiplier: float = 2.0,
        **_kwargs: Any,
    ) -> None:
        """
        默认移动止损实现：ATR × multiplier 止损线只上移不下移。
        同时更新高低价和未实现盈亏。
        """
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price

        if atr > 0.0:
            candidate = self.highest_price - atr * multiplier
            if candidate > self.trailing_stop_price:
                self.trailing_stop_price = candidate

        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity

    # ------------------------------------------------------------------
    # 只读派生属性
    # ------------------------------------------------------------------
    @property
    def pnl_ratio(self) -> float:
        """基于当前未实现盈亏的盈亏率"""
        if self.entry_price <= 0.0:
            return 0.0
        return self.unrealized_pnl / (self.entry_price * self.quantity)

    @property
    def holding_days(self) -> int:
        """持仓天数（相对回测当前日期需外部传入；此处基于 today 作为静态查询）"""
        try:
            entry = datetime.strptime(self.entry_date, "%Y-%m-%d").date()
            return (date.today() - entry).days
        except (ValueError, TypeError):
            return 0

    def holding_days_from(self, current_date: str) -> int:
        """给定回测当前日期精确计算持仓天数"""
        try:
            entry = datetime.strptime(self.entry_date, "%Y-%m-%d").date()
            current = datetime.strptime(current_date, "%Y-%m-%d").date()
            return (current - entry).days
        except (ValueError, TypeError):
            return 0

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code={self.code}, "
            f"entry={self.entry_price:.2f}, pnl={self.pnl_ratio:.2%})"
        )


# ============================================================================
# Part 4b: Mixin 层
# 每个 Mixin 仅声明增量 __slots__，与 BasePositionState 组合时无 __dict__ 泄漏。
# ============================================================================


class TrailingStopMixin:
    """
    ATR 动态移动止损 Mixin。

    覆盖 update_trailing_stop，在 BasePositionState 默认逻辑基础上
    保证止损线严格单调递增（止损只上移）。
    适用于：ShortTermRSRS、MomentumReversal。
    """

    # 本 Mixin 不引入新字段（highest_price / trailing_stop_price 已在 Base 中），
    # 仅提供覆盖方法。
    __slots__ = ()

    def update_trailing_stop(
        self,
        current_price: float,
        atr: float = 0.0,
        multiplier: float = 2.0,
        **_kwargs: Any,
    ) -> None:
        # 类型标注辅助：self 实际是 BasePositionState 子类
        _self: BasePositionState = self  # type: ignore[assignment]

        if current_price > _self.highest_price:
            _self.highest_price = current_price

        if current_price < _self.lowest_price:
            _self.lowest_price = current_price

        if atr > 0.0:
            candidate = _self.highest_price - atr * multiplier
            # 止损线只上移
            if candidate > _self.trailing_stop_price:
                _self.trailing_stop_price = candidate

        _self.unrealized_pnl = (current_price - _self.entry_price) * _self.quantity


class LockProfitMixin:
    """
    分级锁利 Mixin（AlphaHunter 核心持仓逻辑）。

    lock_levels 定义利润率梯度 [3%, 6%, 9%, 12%, 15%]；
    每穿过一个梯度，止损线跳跃上移至对应锁利位。
    与 TrailingStopMixin 可叠加：先执行 LockProfit 更新，再由 TrailingStop 兜底。

    __slots__ 设计说明：
        Python 不允许多个基类同时拥有非空实例布局的 __slots__。
        Mixin 声明 __slots__ = ()；其数据字段由 concrete 叶类在自身 __slots__
        中补充声明。Mixin 方法通过 self 动态访问这些字段。
        叶类需在 __slots__ 中声明：lock_levels, current_lock_level, hard_stop
        并在 __init__ 中调用 self._init_lock_profit(...) 完成初始化。
    """

    __slots__ = ()  # 数据字段由叶类声明

    def _init_lock_profit(
        self,
        lock_levels: Optional[List[float]] = None,
        hard_stop: float = 0.0,
    ) -> None:
        """叶类 __init__ 中显式调用以初始化锁利字段"""
        self.lock_levels = lock_levels if lock_levels is not None else [  # type: ignore[attr-defined]
            0.03, 0.06, 0.09, 0.12, 0.15
        ]
        self.current_lock_level = 0  # type: ignore[attr-defined]
        self.hard_stop = hard_stop  # type: ignore[attr-defined]

    def apply_lock_profit(self, current_price: float) -> None:
        """根据当前价格和入场价计算盈亏率，逐级检查锁利梯度，更新止损线。"""
        if self.entry_price <= 0.0:  # type: ignore[attr-defined]
            return

        pnl = (current_price - self.entry_price) / self.entry_price  # type: ignore[attr-defined]

        while (
            self.current_lock_level < len(self.lock_levels)  # type: ignore[attr-defined]
            and pnl >= self.lock_levels[self.current_lock_level]  # type: ignore[attr-defined]
        ):
            lock_stop = self.entry_price * (1.0 + 0.02 * (self.current_lock_level + 1))  # type: ignore[attr-defined]
            if lock_stop > self.trailing_stop_price:  # type: ignore[attr-defined]
                self.trailing_stop_price = lock_stop  # type: ignore[attr-defined]
            self.current_lock_level += 1  # type: ignore[attr-defined]

        # hard_stop 兜底
        if self.hard_stop > 0.0 and self.hard_stop > self.trailing_stop_price:  # type: ignore[attr-defined]
            self.trailing_stop_price = self.hard_stop  # type: ignore[attr-defined]


class TimeStopMixin:
    """
    时间止损 Mixin（SentimentReversal 核心逻辑）。

    持仓超过 max_holding_days 后强制退场，避免情绪反转信号失效后的长期套牢。
    叶类需在 __slots__ 中声明 max_holding_days，并在 __init__ 中调用 _init_time_stop()。
    """

    __slots__ = ()

    def _init_time_stop(self, max_holding_days: int = 5) -> None:
        """叶类 __init__ 中显式调用以初始化时间止损字段"""
        self.max_holding_days = max_holding_days  # type: ignore[attr-defined]

    def is_time_expired(self, current_date: str) -> bool:
        """判断是否超过最大持仓天数。current_date 由回测引擎传入。"""
        return self.holding_days_from(current_date) >= self.max_holding_days  # type: ignore[attr-defined]


class BarCountMixin:
    """
    K线计数止损 Mixin（UltraShort 专用）。

    用于日内策略，按 K 线根数而非日历天数判断退场。
    叶类需在 __slots__ 中声明 entry_bar_index, bars_held, initial_momentum，
    并在 __init__ 中调用 _init_bar_count()。
    """

    __slots__ = ()

    def _init_bar_count(
        self,
        entry_bar_index: int = 0,
        initial_momentum: float = 0.0,
    ) -> None:
        """叶类 __init__ 中显式调用以初始化 K 线计数字段"""
        self.entry_bar_index = entry_bar_index  # type: ignore[attr-defined]
        self.bars_held = 0  # type: ignore[attr-defined]
        self.initial_momentum = initial_momentum  # type: ignore[attr-defined]

    def tick_bar(self) -> None:
        """每根 K 线结束时调用一次，递增计数器。"""
        self.bars_held += 1  # type: ignore[attr-defined]


# ============================================================================
# Part 5: FactorStore + FactorAccessor
#
# ── Numba 因子引擎对接点 ──
#
# AlphaFactorEngineV2.compute() 返回的核心数据结构是：
#     Dict[str, np.ndarray]   shape = (n_stocks, n_days)  ← C-contiguous, float64
#
# 策略层不应直接操作 ndarray（索引语义不透明，维护噩梦）。
# FactorAccessor 充当翻译层：
#   - 内部持有 ndarray 矩阵引用（零拷贝）
#   - 外部暴露 .get(factor_name, code, date_idx) → float 接口
#   - code → 列索引映射由 code_to_idx: Dict[str, int] 提供（引擎注入）
#   - date_idx = -1 表示最新一日（回测当前时间点）
#
# 性能路径：
#   get() 内部仅做 dict lookup + ndarray[row, col] 索引，无 pandas 开销。
#   对于需要截面级因子值的场景（如 Top-N 排序），提供 get_cross_section()
#   直接返回 ndarray 切片，供 np.argsort 等向量化操作直接使用。
#
# 兼容路径：
#   若引擎尚未切换到 Numba 路径（Phase 3 之前），底层可以是 DataFrame/Series，
#   FactorAccessor 通过 isinstance 自动切换访问逻辑，策略代码无需修改。
# ============================================================================

# 类型别名
FactorStore = Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]]
"""
FactorStore 的值可以是三种形式之一：
  pd.DataFrame  — index=date, columns=code（传统路径）
  pd.Series     — index=code（截面因子快速查询）
  np.ndarray    — shape=(n_stocks, n_days)（Numba 引擎输出，高性能路径）
"""


class FactorAccessor:
    """
    因子访问器——策略层与因子引擎之间的唯一桥接。

    屏蔽底层存储差异（DataFrame / Series / ndarray），
    暴露统一的 .get() 和 .get_cross_section() 接口。

    线程安全说明：
      FactorAccessor 本身只读；store 由引擎在每个时间步开始前注入一次，
      策略内 generate_signals 期间不会被修改。因此可安全在同一线程内多次调用。
    """

    __slots__ = ("_store", "_code_to_idx", "_date_to_idx", "_current_date_idx")

    def __init__(
        self,
        store: FactorStore,
        code_to_idx: Optional[Dict[str, int]] = None,
        date_to_idx: Optional[Dict[str, int]] = None,
        current_date_idx: int = -1,
    ) -> None:
        """
        Args:
            store:            因子存储字典
            code_to_idx:      股票代码→矩阵列索引映射（ndarray 路径必须提供）
            date_to_idx:      日期字符串→矩阵行索引映射（ndarray 路径必须提供）
            current_date_idx: 回测当前日期对应的行索引（-1 表示最后一行）
        """
        self._store = store
        self._code_to_idx: Dict[str, int] = code_to_idx or {}
        self._date_to_idx: Dict[str, int] = date_to_idx or {}
        self._current_date_idx = current_date_idx

    # ------------------------------------------------------------------
    # 核心查询接口
    # ------------------------------------------------------------------
    def get(
        self,
        factor_name: str,
        code: str,
        date: Optional[str] = None,
    ) -> Optional[float]:
        """
        获取单个因子值。

        Args:
            factor_name: 因子名，如 "rsrs_score", "r2", "atr"
            code:        股票代码
            date:        日期字符串；None 表示当前回测日期
        Returns:
            因子值 float；不存在时返回 None。
        """
        raw = self._store.get(factor_name)
        if raw is None:
            return None

        # --- 路径 A：ndarray（Numba 高性能路径） ---
        if isinstance(raw, np.ndarray):
            col = self._code_to_idx.get(code)
            if col is None:
                return None
            if date is None:
                row = self._current_date_idx
            else:
                row = self._date_to_idx.get(date)
                if row is None:
                    return None
            if raw.ndim == 2:
                if 0 <= row < raw.shape[0] and 0 <= col < raw.shape[1]:
                    val = raw[row, col]
                    return None if np.isnan(val) else float(val)
                # 支持负数索引（-1 = 最后一行）
                if row < 0 and (-row) <= raw.shape[0] and 0 <= col < raw.shape[1]:
                    val = raw[row, col]
                    return None if np.isnan(val) else float(val)
            return None

        # --- 路径 B：DataFrame（传统路径） ---
        if isinstance(raw, pd.DataFrame):
            lookup_date = date if date else None
            if lookup_date and lookup_date in raw.index and code in raw.columns:
                val = raw.loc[lookup_date, code]
                return None if pd.isna(val) else float(val)
            # date=None 时取最后一行
            if lookup_date is None and code in raw.columns:
                series = raw[code].dropna()
                if len(series) > 0:
                    return float(series.iloc[-1])
            return None

        # --- 路径 C：Series（截面快速查询） ---
        if isinstance(raw, pd.Series):
            if code in raw.index:
                val = raw[code]
                return None if pd.isna(val) else float(val)
            return None

        return None

    def get_cross_section(
        self,
        factor_name: str,
        date: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        获取截面因子向量（所有股票在某日期的因子值）。
        返回 shape=(n_stocks,) 的 ndarray；用于 Top-N 排序等向量化操作。

        ndarray 路径下为零拷贝视图；DataFrame 路径下会产生一次 .values 拷贝。
        """
        raw = self._store.get(factor_name)
        if raw is None:
            return None

        if isinstance(raw, np.ndarray) and raw.ndim == 2:
            row = self._current_date_idx if date is None else self._date_to_idx.get(date, -1)
            if row < 0 and (-row) > raw.shape[0]:
                return None
            return raw[row, :]  # 视图，无拷贝

        if isinstance(raw, pd.DataFrame):
            if date and date in raw.index:
                return raw.loc[date].values.astype(np.float64)
            if date is None and len(raw) > 0:
                return raw.iloc[-1].values.astype(np.float64)
            return None

        if isinstance(raw, pd.Series):
            return raw.values.astype(np.float64)

        return None

    # ------------------------------------------------------------------
    # 批量查询
    # ------------------------------------------------------------------
    def get_multi(
        self,
        factor_name: str,
        codes: List[str],
        date: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """对多个股票批量查询同一因子，返回 {code: value}"""
        return {code: self.get(factor_name, code, date) for code in codes}

    def has_factor(self, factor_name: str) -> bool:
        """检查因子是否存在于 store 中"""
        return factor_name in self._store

    def factor_names(self) -> List[str]:
        """返回当前 store 中所有因子名"""
        return list(self._store.keys())

    def __repr__(self) -> str:
        return f"FactorAccessor(factors={self.factor_names()}, codes={len(self._code_to_idx)})"


# ============================================================================
# Part 6: StrategyContext — 引擎注入策略的运行上下文
#
# 设计要点：
#   - current_data 是当日全市场行情 DataFrame（单日快照）
#   - _history_provider / _factor_provider 由引擎以闭包形式注入，
#     策略层无需 import 数据库或数据管理模块 → 六边形架构隔离
#   - factor_accessor 是 FactorAccessor 实例，策略优先使用此接口查询因子
# ============================================================================


class StrategyContext:
    """
    策略运行上下文（由 HybridExecutionEngine 每时间步构建并传入策略）。

    使用 __slots__ 避免动态属性分配；引擎每日创建一个新实例，代价极低。
    """

    __slots__ = (
        "current_date",
        "current_timestamp",
        "current_data",
        "positions",          # {code: quantity}  引擎维护的持仓快照
        "total_equity",
        "cash",
        "universe",           # List[str] 股票池列表
        # 闭包提供者（引擎注入，策略不可见）
        "_history_provider",
        "_factor_provider",
        # 因子访问器
        "factor_accessor",
    )

    def __init__(
        self,
        current_date: str,
        current_data: pd.DataFrame,
        positions: Dict[str, int],
        total_equity: float,
        cash: float,
        current_timestamp: Optional[Timestamp] = None,
        history_provider: Optional[Callable[[str, int], pd.DataFrame]] = None,
        factor_provider: Optional[Callable[[str, str], Optional[float]]] = None,
        factor_accessor: Optional[FactorAccessor] = None,
        universe: Optional[List[str]] = None,
    ) -> None:
        self.current_date = current_date
        self.current_timestamp = current_timestamp
        self.current_data = current_data
        self.positions = positions
        self.total_equity = total_equity
        self.cash = cash
        self.universe = universe if universe is not None else []
        self._history_provider = history_provider
        self._factor_provider = factor_provider
        self.factor_accessor = factor_accessor

    # ------------------------------------------------------------------
    # 数据访问接口
    # ------------------------------------------------------------------
    def get_history(self, code: str, lookback: int = 250) -> pd.DataFrame:
        """获取股票历史 K 线数据。由引擎提供者实现，策略层透明调用。"""
        if self._history_provider is not None:
            return self._history_provider(code, lookback)
        return pd.DataFrame()

    def get_factor(self, factor_name: str, code: str) -> Optional[float]:
        """
        获取因子值。优先走 FactorAccessor（支持 ndarray 高性能路径）；
        回退到 _factor_provider 闭包（兼容旧路径）。
        """
        if self.factor_accessor is not None:
            return self.factor_accessor.get(factor_name, code)
        if self._factor_provider is not None:
            return self._factor_provider(factor_name, code)
        return None

    # ------------------------------------------------------------------
    # 市场状态汇总
    # ------------------------------------------------------------------
    def get_market_state(self) -> Dict[str, Any]:
        """
        构建当日市场状态快照。
        涨跌比、成仓数、权益等信息供策略 _check_market_condition 使用。
        """
        if self.current_data.empty:
            return {
                "date": self.current_date,
                "total_stocks": 0,
                "advancing": 0,
                "declining": 0,
                "unchanged": 0,
                "advance_ratio": 0.5,
                "positions_count": 0,
                "total_equity": self.total_equity,
                "cash": self.cash,
            }

        df = self.current_data
        if "close" in df.columns and "open" in df.columns:
            changes = df["close"] / df["open"] - 1.0
            advancing = int((changes > 0.001).sum())
            declining = int((changes < -0.001).sum())
            unchanged = len(df) - advancing - declining
        else:
            advancing = declining = unchanged = 0

        total = len(df)
        return {
            "date": self.current_date,
            "total_stocks": total,
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "advance_ratio": advancing / total if total > 0 else 0.5,
            "positions_count": len([q for q in self.positions.values() if q > 0]),
            "total_equity": self.total_equity,
            "cash": self.cash,
        }

    def get_position_weight(self, code: str) -> float:
        """获取单股持仓权重（相对于总权益）"""
        if self.total_equity <= 0.0:
            return 0.0
        qty = self.positions.get(code, 0)
        if qty <= 0:
            return 0.0
        if not self.current_data.empty and "code" in self.current_data.columns:
            row = self.current_data[self.current_data["code"] == code]
            if not row.empty and "close" in row.columns:
                price = float(row["close"].iloc[0])
                return (qty * price) / self.total_equity
        return 0.0

    def __repr__(self) -> str:
        return (
            f"StrategyContext(date={self.current_date}, "
            f"positions={len(self.positions)}, equity={self.total_equity:.2f})"
        )


# ============================================================================
# Part 7: 异常体系
#
# 5 层封顶，精确捕捉。引擎层 catch StrategyException 统一处理；
# 策略层可 raise 具体子类表达语义。
# ============================================================================


class StrategyException(Exception):
    """策略层异常根类"""

    def __init__(self, message: str, strategy_name: str = "", code: str = "") -> None:
        self.strategy_name = strategy_name
        self.code = code
        super().__init__(
            f"[{strategy_name}] {message}" if strategy_name else message
        )


class StrategyConfigError(StrategyException):
    """参数配置校验失败（如 top_n < 0）"""


class FactorComputeError(StrategyException):
    """因子计算过程中抛出异常（如数据不足、除零）"""


class SignalGenerationError(StrategyException):
    """信号生成阶段异常（如 weight 越界、code 格式错误）"""


class PositionStateError(StrategyException):
    """持仓状态异常（如对不存在的仓位操作）"""


class RiskLimitExceeded(StrategyException):
    """风控限制触发（如单股权重超限、总仓位超限）"""


# ============================================================================
# Part 8: 导出
# ============================================================================

__all__ = [
    # 枚举
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Timeframe",
    "StrategyState",
    # 时间
    "Timestamp",
    # 交易流水线
    "Signal",
    "Order",
    "Fill",
    # 持仓协议 + 基础实现
    "PositionStateProtocol",
    "BasePositionState",
    # Mixin
    "TrailingStopMixin",
    "LockProfitMixin",
    "TimeStopMixin",
    "BarCountMixin",
    # 因子
    "FactorStore",
    "FactorAccessor",
    # 上下文
    "StrategyContext",
    # 异常
    "StrategyException",
    "StrategyConfigError",
    "FactorComputeError",
    "SignalGenerationError",
    "PositionStateError",
    "RiskLimitExceeded",
]
