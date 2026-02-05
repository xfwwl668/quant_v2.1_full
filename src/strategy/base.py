"""
strategy/base.py
================
Phase 2 — 策略协议层：BaseStrategy 抽象基类。

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. 双层参数注入 (Dual-Layer Parameter Injection)
   ─────────────────────────────────────────────
   层 A — ClassVar DEFAULT_PARAMS：策略类级别的默认参数字典。
   层 B — __init__(params)：实例级别的覆盖参数。
          params 可以是 Dict 或 任意 dataclass 实例（如 MRConfig、SentimentConfig）；
          dataclass 会被自动展平为 Dict，再与 DEFAULT_PARAMS 做 shallow merge。
   合并优先级：  instance params  >  DEFAULT_PARAMS

   目的：解决现有 5 策略中 Dict vs Dataclass 参数格式混战的问题，
   让子策略无需感知参数来源，统一通过 get_param() 访问。

2. 模板方法流水线 (Template Method Pipeline)
   ──────────────────────────────────────────
   generate_signals() 是 sealed 方法（子类不应重写），它驱动三级流水线：

       ┌─────────────────────────────────────────┐
       │  Step 1: _generate_exit_signals(ctx)     │  退仓信号（止损/止盈/时间止损）
       │      ↓                                   │
       │  Step 2: _check_market_condition(ctx)    │  市场环境门槛（跌停板数/涨跌比等）
       │      ↓ (False → 仅返回 exit signals)     │
       │  Step 3: _generate_entry_signals(ctx)    │  入仓信号（核心 alpha 逻辑）
       │      ↓                                   │
       │  return exit_signals + entry_signals     │
       └─────────────────────────────────────────┘

   子类必须实现：_generate_entry_signals（核心差异化逻辑）
   子类可选覆盖：_generate_exit_signals、_check_market_condition

3. FactorStore ←→ Numba 因子引擎对接路径
   ─────────────────────────────────────
   ┌──────────────────┐     ┌────────────────────┐
   │ AlphaFactorEngV2 │────▶│  ndarray matrices  │  shape=(n_stocks, n_days)
   │   (Numba @njit)  │     │  C-contiguous f64  │
   └──────────────────┘     └────────┬───────────┘
                                     │  零拷贝引用
                                     ▼
                            ┌────────────────────┐
                            │   FactorAccessor   │  ← 策略层唯一入口
                            │  .get(name, code)  │     dict lookup + arr[r,c]
                            └────────┬───────────┘
                                     │  注入为
                                     ▼
                            ┌────────────────────┐
                            │  StrategyContext   │
                            │  .factor_accessor  │
                            └────────────────────┘

   策略中调用 ctx.get_factor("rsrs_score", "000001") 时，
   底层路径是 ndarray 索引查找（~50ns），无 pandas 开销。
   对于截面级 Top-N 排序，用 ctx.factor_accessor.get_cross_section("rsrs_score")
   直接返回 ndarray 切片，交给 np.argsort 处理。

4. __slots__ 策略
   ────────────
   BaseStrategy 不用 __slots__（它持有 logger、deque 等复合对象，
   且实例数量极少——通常不超过 10 个策略并行）。
   热路径对象（Signal, Order, Fill, PositionState）均已在 types.py 中开启 __slots__。

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import dataclasses
import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import (
    Any,
    ClassVar,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from .types import (
    FactorAccessor,
    FactorStore,
    Fill,
    Order,
    OrderSide,
    PositionStateProtocol,
    Signal,
    StrategyConfigError,
    StrategyContext,
    StrategyException,
    StrategyState,
    Timeframe,
)


# ============================================================================
# BaseStrategy
# ============================================================================


class BaseStrategy(ABC):
    """
    策略抽象基类 — 所有量化策略的统一接口与流程控制器。

    子类必须实现的抽象方法：
        compute_factors(history) → FactorStore
        _generate_entry_signals(context) → List[Signal]

    子类可选覆盖的钩子：
        initialize()                          一次性初始化
        _generate_exit_signals(context)       退仓信号生成（默认遍历持仓检查止损）
        _check_market_condition(context)      市场环境门槛（默认 True）
        on_bar(context)                       每根 K 线回调（用于实时止损）
        on_order_filled(order)                成交回调
        on_order_rejected(order, reason)      拒绝回调
        on_day_start(context)                 日开回调
        on_day_end(context)                   日末回调
        on_strategy_start()                   策略启动回调
        on_strategy_stop()                    策略停止回调
    """

    # ------------------------------------------------------------------
    # 类级别声明（子类必须覆盖 name）
    # ------------------------------------------------------------------
    name: ClassVar[str] = "base_strategy"
    version: ClassVar[str] = "1.0.0"
    timeframe: ClassVar[Timeframe] = Timeframe.D1

    # 默认参数字典 — 层 A。子类定义此 ClassVar 即可提供策略级别默认值。
    DEFAULT_PARAMS: ClassVar[Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # __init__ — 双层参数注入入口
    # ------------------------------------------------------------------
    def __init__(self, params: Optional[Union[Dict[str, Any], Any]] = None) -> None:
        """
        双层参数注入：
            层 A: self.DEFAULT_PARAMS（ClassVar，子类定义）
            层 B: params（实例传入，Dict 或 dataclass）
        合并规则: {**DEFAULT_PARAMS, **flattened(params)}

        Args:
            params: 覆盖参数。
                    - None        → 仅使用 DEFAULT_PARAMS
                    - Dict        → 直接合并
                    - dataclass   → dataclasses.asdict() 展平后合并
        """
        self._params: Dict[str, Any] = dict(self.DEFAULT_PARAMS)  # 层 A 拷贝

        if params is not None:
            if isinstance(params, dict):
                self._params.update(params)  # 层 B: Dict 路径
            elif dataclasses.is_dataclass(params) and not isinstance(params, type):
                # 层 B: dataclass 实例路径（如 MRConfig()、SentimentConfig()）
                self._params.update(dataclasses.asdict(params))
            else:
                raise StrategyConfigError(
                    f"params 必须是 Dict 或 dataclass 实例，得到 {type(params).__name__}",
                    strategy_name=self.name,
                )

        # 日志器
        self.logger: logging.Logger = logging.getLogger(f"strategy.{self.name}")

        # 生命周期状态
        self._state: StrategyState = StrategyState.CREATED

        # 持仓状态映射：{code → PositionStateProtocol 实现}
        # 引擎层仅依赖 Protocol 接口，不感知具体类型。
        self._positions: Dict[str, PositionStateProtocol] = {}

        # 交易历史（滚动窗口，避免无限增长）
        self._trade_history: Deque[Dict[str, Any]] = deque(maxlen=2000)

        # 因子缓存 — compute_factors 结果暂存，供本策略 generate_signals 期间复用
        self._factor_cache: Dict[str, Any] = {}

        # 调用子类初始化钩子
        self.initialize()
        self._state = StrategyState.INITIALIZED
        self.logger.info(f"Strategy '{self.name}' v{self.version} initialized | params={self._params}")

    # ==================== 参数管理 ====================

    def get_param(self, key: str, default: Any = None) -> Any:
        """获取参数值（统一入口，屏蔽层 A / 层 B 来源差异）"""
        return self._params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """运行期动态修改参数（如网格搜索时）"""
        self._params[key] = value

    def get_params(self) -> Dict[str, Any]:
        """获取参数快照（返回拷贝，防外部污染）"""
        return dict(self._params)

    # ==================== 抽象方法 ====================

    @abstractmethod
    def compute_factors(
        self, history: Dict[str, pd.DataFrame]
    ) -> FactorStore:
        """
        预计算因子（回测开始前或定期调用）。

        ── Numba 对接路径注释 ──
        此方法返回的 FactorStore 会被引擎包装为 FactorAccessor 注入 StrategyContext。
        如果子策略希望利用 Numba 加速路径，可以在此处调用
        AlphaFactorEngineV2.compute() 获取 ndarray，直接放入返回的 dict；
        引擎会自动识别 ndarray 类型并构建 code_to_idx 映射。

        示例：
            def compute_factors(self, history):
                # 传统路径（兼容）
                rsrs_df = self._calc_rsrs(history)        # → DataFrame
                # 高性能路径（Phase 3 之后）
                # rsrs_arr = AlphaFactorEngineV2.compute_rsrs(history_matrix)  # → ndarray
                return {"rsrs_score": rsrs_df}

        Args:
            history: {code: OHLCV DataFrame}
        Returns:
            FactorStore = {factor_name: DataFrame | Series | ndarray}
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_entry_signals(self, context: StrategyContext) -> List[Signal]:
        """
        入仓信号生成（策略核心 alpha 逻辑）。

        由 generate_signals 流水线在通过退仓检查和市场环境门槛后调用。
        子类必须实现此方法。

        典型流程：
            1. 从 context.factor_accessor 获取因子值
            2. 构建评分/排名
            3. 筛选 Top-N，生成 Signal 列表

        Args:
            context: 当前时间步的策略上下文
        Returns:
            入仓信号列表
        """
        raise NotImplementedError

    # ==================== 可选钩子 ====================

    def initialize(self) -> None:
        """
        初始化钩子（在 __init__ 末尾调用一次）。

        用于：加载外部数据、初始化模型、创建策略专用容器等。
        默认空实现，子类按需覆盖。
        """
        self.logger.debug(f"initialize() called for '{self.name}'")

    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """
        退仓信号生成（默认实现）。

        遍历当前所有持仓，检查通用退场条件：
            - 止损价 (stop_loss_price)
            - 止盈价 (take_profit_price)
            - 移动止损价 (trailing_stop_price)
        子类可完全覆盖此方法以实现自定义退场逻辑
        （如时间止损、K 线计数止损等由 Mixin 提供）。

        Returns:
            退仓 Signal 列表（weight=0.0 表示清仓）
        """
        exit_signals: List[Signal] = []
        df = context.current_data

        for code, pos in list(self._positions.items()):
            # 获取当前价格
            current_price = self._get_current_price(code, df)
            if current_price is None:
                continue

            # 更新移动止损（由 PositionState 实现）
            pos.update_trailing_stop(current_price)

            # 检查止损
            if hasattr(pos, "stop_loss_price") and pos.stop_loss_price > 0.0:  # type: ignore[attr-defined]
                if current_price <= pos.stop_loss_price:  # type: ignore[attr-defined]
                    exit_signals.append(Signal(
                        code=code,
                        side=OrderSide.SELL,
                        weight=0.0,
                        reason=f"止损触发 price={current_price:.2f} <= stop={pos.stop_loss_price:.2f}",  # type: ignore[attr-defined]
                        strategy_name=self.name,
                    ))
                    continue  # 已退场，不再检查其他条件

            # 检查止盈
            if hasattr(pos, "take_profit_price") and pos.take_profit_price < math.inf:  # type: ignore[attr-defined]
                if current_price >= pos.take_profit_price:  # type: ignore[attr-defined]
                    exit_signals.append(Signal(
                        code=code,
                        side=OrderSide.SELL,
                        weight=0.0,
                        reason=f"止盈触发 price={current_price:.2f} >= tp={pos.take_profit_price:.2f}",  # type: ignore[attr-defined]
                        strategy_name=self.name,
                    ))
                    continue

            # 检查移动止损
            if hasattr(pos, "trailing_stop_price") and pos.trailing_stop_price > 0.0:  # type: ignore[attr-defined]
                if current_price <= pos.trailing_stop_price:  # type: ignore[attr-defined]
                    exit_signals.append(Signal(
                        code=code,
                        side=OrderSide.SELL,
                        weight=0.0,
                        reason=f"移动止损触发 price={current_price:.2f} <= trail={pos.trailing_stop_price:.2f}",  # type: ignore[attr-defined]
                        strategy_name=self.name,
                    ))
                    continue

            # 检查 LockProfitMixin：若存在 apply_lock_profit，调用更新锁利止损
            if hasattr(pos, "apply_lock_profit"):
                pos.apply_lock_profit(current_price)  # type: ignore[attr-defined]
                # 更新后再次检查 trailing_stop_price
                if hasattr(pos, "trailing_stop_price") and pos.trailing_stop_price > 0.0:  # type: ignore[attr-defined]
                    if current_price <= pos.trailing_stop_price:  # type: ignore[attr-defined]
                        exit_signals.append(Signal(
                            code=code,
                            side=OrderSide.SELL,
                            weight=0.0,
                            reason=f"锁利止损触发 price={current_price:.2f} <= lock_trail={pos.trailing_stop_price:.2f}",  # type: ignore[attr-defined]
                            strategy_name=self.name,
                        ))
                        continue

            # 检查 TimeStopMixin：时间止损
            if hasattr(pos, "is_time_expired"):
                if pos.is_time_expired(context.current_date):  # type: ignore[attr-defined]
                    exit_signals.append(Signal(
                        code=code,
                        side=OrderSide.SELL,
                        weight=0.0,
                        reason=f"时间止损触发 holding >= max_days={pos.max_holding_days}",  # type: ignore[attr-defined]
                        strategy_name=self.name,
                    ))
                    continue

            # 检查 BarCountMixin：K线计数止损（日内策略）
            if hasattr(pos, "bars_held") and hasattr(pos, "tick_bar"):
                pos.tick_bar()  # type: ignore[attr-defined]

        return exit_signals

    def _check_market_condition(self, context: StrategyContext) -> bool:
        """
        市场环境门槛检查。

        默认返回 True（不做市场过滤）。
        子类可覆盖此方法实现如下逻辑：
            - 当日跌停板数 > N → 暂停入仓（大跌日避险）
            - 涨跌比 < 0.3 → 市场情绪极度悲观 → 仅保留退仓信号

        通常利用 context.get_market_state() 获取涨跌比等指标。
        """
        return True

    def on_bar(self, context: StrategyContext) -> List[Signal]:
        """
        每根 K 线回调（用于实时止损止盈，主要面向日内策略）。

        日线策略一般不需要覆盖此方法（退仓逻辑已在 _generate_exit_signals 中处理）。
        日内策略（UltraShort 等）可在此处检查分钟级止损。

        Returns:
            额外的交易信号
        """
        return []

    def on_order_filled(self, order: Order) -> None:
        """
        订单成交回调。

        默认实现：自动注册 BUY 成交为 BasePositionState，
        自动注销 SELL 成交并记录交易历史。

        子策略若需要创建自定义 PositionState（如 AlphaHunterPosition），
        应覆盖此方法，在 BUY 分支中构造对应子类实例并调用 register_position()。
        """
        if order.side == OrderSide.BUY:
            # 默认注册为 BasePositionState
            from .types import BasePositionState
            stop_loss_rate = self.get_param("stop_loss", -0.08)
            take_profit_rate = self.get_param("take_profit", 0.15)

            pos = BasePositionState(
                code=order.code,
                entry_price=order.filled_price,
                entry_date=order.create_date,
                quantity=order.filled_quantity,
                stop_loss_price=order.filled_price * (1.0 + stop_loss_rate),  # stop_loss 为负值
                take_profit_price=order.filled_price * (1.0 + take_profit_rate),
            )
            self.register_position(pos)
            self.logger.info(
                f"[BUY FILLED] {order.code} @ {order.filled_price:.2f} "
                f"qty={order.filled_quantity} | SL={pos.stop_loss_price:.2f} TP={pos.take_profit_price:.2f}"
            )

        elif order.side == OrderSide.SELL:
            old_pos = self.unregister_position(order.code)
            if old_pos is not None:
                pnl_ratio = (order.filled_price - old_pos.entry_price) / old_pos.entry_price
                self.record_trade({
                    "code": order.code,
                    "entry_date": old_pos.entry_date,
                    "exit_date": order.create_date,
                    "entry_price": old_pos.entry_price,
                    "exit_price": order.filled_price,
                    "quantity": old_pos.quantity,
                    "pnl_ratio": pnl_ratio,
                    "is_win": pnl_ratio > 0,
                    "strategy_name": self.name,
                })
                self.logger.info(
                    f"[SELL FILLED] {order.code} @ {order.filled_price:.2f} "
                    f"PnL={pnl_ratio:.2%} | entry={old_pos.entry_date}"
                )
            else:
                self.logger.warning(f"[SELL FILLED] {order.code} — 无对应持仓记录")

    def on_order_rejected(self, order: Order, reason: str) -> None:
        """订单拒绝回调"""
        self.logger.warning(f"[REJECTED] {order.code} {order.side.value}: {reason}")

    def on_day_start(self, context: StrategyContext) -> None:
        """日开回调（默认空）"""
        self.logger.debug(f"on_day_start: {context.current_date}")

    def on_day_end(self, context: StrategyContext) -> None:
        """日末回调（默认空）"""
        self.logger.debug(f"on_day_end: {context.current_date} | positions={len(self._positions)}")

    def on_strategy_start(self) -> None:
        """策略启动回调（由 Backtester 在回测开始时调用）"""
        self._state = StrategyState.RUNNING
        self.logger.info(f"Strategy '{self.name}' started")

    def on_strategy_stop(self) -> None:
        """策略停止回调（由 Backtester 在回测结束时调用）"""
        self._state = StrategyState.STOPPED
        self.logger.info(f"Strategy '{self.name}' stopped")

    # ==================== 模板方法：信号生成流水线 ====================
    # generate_signals 是密封方法，子类不应重写。
    # 流水线：退仓 → 市场环境检查 → 入仓

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """
        信号生成流水线入口（密封，不应被子类覆盖）。

        执行顺序：
            1. _generate_exit_signals  → 退仓信号（高优先级，先执行）
            2. _check_market_condition → 市场环境门槛
               若不通过 → 仅返回退仓信号（不入仓）
            3. _generate_entry_signals → 入仓信号

        Returns:
            合并后的信号列表；退仓信号在前（引擎按此顺序处理，先平后开）
        """
        # Step 1: 退仓信号
        exit_signals = self._generate_exit_signals(context)

        # Step 2: 市场环境门槛
        if not self._check_market_condition(context):
            self.logger.debug(f"[{context.current_date}] 市场环境检查未通过，仅执行退仓")
            return exit_signals

        # Step 3: 入仓信号
        entry_signals = self._generate_entry_signals(context)

        return exit_signals + entry_signals

    # ==================== 持仓管理 ====================

    def register_position(self, state: PositionStateProtocol) -> None:
        """注册持仓状态。引擎层通过 Protocol 接口调用，无需知道具体类型。"""
        self._positions[state.code] = state

    def unregister_position(self, code: str) -> Optional[PositionStateProtocol]:
        """注销持仓状态，返回被移除的状态对象（用于记录交易）"""
        return self._positions.pop(code, None)

    def get_position(self, code: str) -> Optional[PositionStateProtocol]:
        """获取单股持仓状态"""
        return self._positions.get(code)

    def get_all_positions(self) -> Dict[str, PositionStateProtocol]:
        """获取所有持仓状态的快照（返回拷贝）"""
        return dict(self._positions)

    @property
    def position_codes(self) -> Set[str]:
        """当前持仓股票代码集合"""
        return set(self._positions.keys())

    @property
    def position_count(self) -> int:
        """当前持仓数"""
        return len(self._positions)

    # ==================== 因子缓存 ====================
    # 策略层缓存 — 与 FactorAccessor（引擎层缓存）独立。
    # 用于策略自身计算的中间因子，如自定义评分、复合指标等。

    def cache_factors(self, factors: Dict[str, Any]) -> None:
        """写入策略级别因子缓存"""
        self._factor_cache.update(factors)

    def get_cached_factor(
        self,
        name: str,
        code: str,
        date_idx: int = -1,
    ) -> Optional[float]:
        """
        从策略级别缓存中读取因子值。

        ── Numba 对接注释 ──
        若缓存中的值是 ndarray（来自 AlphaFactorEngineV2），
        date_idx=-1 表示当前回测日期对应行；
        引擎注入 FactorAccessor 时会设置 current_date_idx。
        此处兼容处理，确保策略层代码无需感知底层容器类型。

        Args:
            name:     因子名
            code:     股票代码
            date_idx: 行索引（-1 为最新）
        Returns:
            因子值 float；不存在返回 None
        """
        data = self._factor_cache.get(name)
        if data is None:
            return None

        try:
            if isinstance(data, np.ndarray):
                # ndarray 路径：需要 code→col 映射
                # 此处假设映射存在于 _factor_cache["__code_to_idx__"]
                code_map: Optional[Dict[str, int]] = self._factor_cache.get("__code_to_idx__")
                if code_map is None:
                    return None
                col = code_map.get(code)
                if col is None:
                    return None
                val = float(data[date_idx, col]) if data.ndim == 2 else float(data[col])
                return None if np.isnan(val) else val

            if isinstance(data, pd.DataFrame):
                if code in data.columns:
                    series = data[code].dropna()
                    if len(series) > 0:
                        return float(series.iloc[date_idx])

            if isinstance(data, pd.Series):
                if code in data.index:
                    val = data[code]
                    return None if pd.isna(val) else float(val)

            if isinstance(data, dict):
                return data.get(code)
        except (IndexError, KeyError, TypeError, ValueError):
            return None

        return None

    # ==================== 交易记录 ====================

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """记录一笔完成的交易"""
        self._trade_history.append(trade)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """获取交易历史快照"""
        return list(self._trade_history)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        计算策略绩效汇总指标。

        Returns:
            包含 win_rate, avg_pnl, profit_factor 等的字典
        """
        trades = list(self._trade_history)
        if not trades:
            return {"trades": 0}

        pnl_ratios = [t["pnl_ratio"] for t in trades if "pnl_ratio" in t]
        if not pnl_ratios:
            return {"trades": len(trades)}

        wins = [p for p in pnl_ratios if p > 0]
        losses = [p for p in pnl_ratios if p <= 0]
        total_win = sum(wins)
        total_loss = sum(losses)  # ≤ 0

        return {
            "trades": len(trades),
            "win_rate": len(wins) / len(pnl_ratios),
            "avg_pnl": float(np.mean(pnl_ratios)),
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "max_win": float(max(pnl_ratios)),
            "max_loss": float(min(pnl_ratios)),
            "total_return": float(total_win + total_loss),
            "profit_factor": (
                abs(total_win / total_loss) if total_loss != 0.0 else math.inf
            ),
        }

    # ==================== 向量化工具 ====================

    @staticmethod
    def vectorized_filter(
        df: pd.DataFrame,
        price_range: Optional[Tuple[float, float]] = None,
        min_amount: Optional[float] = None,
        min_volume: Optional[float] = None,
        exclude_codes: Optional[Set[str]] = None,
        include_codes: Optional[Set[str]] = None,
        factor_conditions: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        统一向量化过滤器 — 提取 5 策略中重复的筛选逻辑。

        所有条件通过布尔 mask 串联，单次 pass 完成过滤，无 for-loop。

        Args:
            df:               当日全市场行情 DataFrame
            price_range:      价格区间 (min, max)
            min_amount:       最小成交额（自动兼容 amount 列和 close*volume 计算）
            min_volume:       最小成交量
            exclude_codes:    排除集（已持仓股票等）
            include_codes:    白名单集（仅在特定池内筛选时使用）
            factor_conditions: 因子条件 {"rsrs_score": (">", 0.7), "r2": (">=", 0.8)}

        Returns:
            过滤后的 DataFrame
        """
        mask = pd.Series(True, index=df.index)

        # 价格区间
        if price_range is not None and "close" in df.columns:
            mask &= df["close"].between(price_range[0], price_range[1])

        # 成交额
        if min_amount is not None:
            if "amount" in df.columns:
                mask &= df["amount"] >= min_amount
            elif "close" in df.columns and "volume" in df.columns:
                mask &= (df["close"] * df["volume"]) >= min_amount

        # 成交量
        if min_volume is not None:
            vol_col = "volume" if "volume" in df.columns else "vol"
            if vol_col in df.columns:
                mask &= df[vol_col] >= min_volume

        # 代码排除
        if exclude_codes and "code" in df.columns:
            mask &= ~df["code"].isin(exclude_codes)

        # 代码白名单
        if include_codes and "code" in df.columns:
            mask &= df["code"].isin(include_codes)

        # 因子条件（要求 df 中已包含对应因子列）
        if factor_conditions:
            _OPS = {
                ">": lambda col, v: col > v,
                ">=": lambda col, v: col >= v,
                "<": lambda col, v: col < v,
                "<=": lambda col, v: col <= v,
                "==": lambda col, v: col == v,
                "!=": lambda col, v: col != v,
            }
            for factor_name, (op, threshold) in factor_conditions.items():
                if factor_name in df.columns and op in _OPS:
                    mask &= _OPS[op](df[factor_name], threshold)

        return df.loc[mask]

    # ==================== 内部工具 ====================

    @staticmethod
    def _get_current_price(code: str, current_data: pd.DataFrame) -> Optional[float]:
        """从当日行情 DataFrame 中提取指定股票的收盘价"""
        if current_data.empty:
            return None
        if "code" in current_data.columns:
            row = current_data.loc[current_data["code"] == code, "close"]
            if not row.empty:
                return float(row.iloc[0])
        elif code in current_data.columns and "close" in current_data.index:
            return float(current_data.loc["close", code])
        return None

    # ==================== 状态查询 ====================

    @property
    def state(self) -> StrategyState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == StrategyState.RUNNING

    # ==================== 表示 ====================

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', version='{self.version}', "
            f"state={self._state.name}, positions={len(self._positions)})"
        )
