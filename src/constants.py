"""
src/constants.py
================
系统全局常量定义（v2.0.1 架构标准）

原则：
  1. 所有魔法数字必须在此定义
  2. 契约常量使用 UPPER_CASE 命名
  3. 配置常量使用 Config 类封装
  4. 避免在业务代码中硬编码

契约红线：
  - TDX_BATCH_SIZE = 800（TDX API 最优批量）
  - VALID_PREFIXES_* 必须与 collector.py 一致
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


# ============================================================================
# Part 1: 市场代码常量（契约级别，不可修改）
# ============================================================================

class Market(Enum):
    """市场代码枚举"""
    SHENZHEN = 0  # 深圳
    SHANGHAI = 1  # 上海


# 有效股票代码前缀（契约）
VALID_PREFIXES_SZ: Tuple[str, ...] = ('00', '30', '8', '43')  # 深圳: 主板/创业板/北交所/新三板精选
VALID_PREFIXES_SH: Tuple[str, ...] = ('60', '688')           # 上海: 主板/科创板


# ============================================================================
# Part 2: 数据采集常量
# ============================================================================

# TDX API 配置
TDX_BATCH_SIZE: int = 800           # TDX 单次获取最大记录数（最优值）
TDX_TIMEOUT: float = 5.0            # 连接超时（秒）
TDX_MAX_FAIL_COUNT: int = 5         # 节点最大失败次数
TDX_NODE_TEST_WORKERS: int = 30     # 节点测速并发数

# 数据采集配置
COLLECTOR_MAX_WORKERS: int = 15     # 下载器最大工作线程数
COLLECTOR_BATCH_LOG_SIZE: int = 100  # 批量日志间隔
COLLECTOR_ASYNC_LOG_QUEUE_SIZE: int = 10000  # 异步日志队列大小


# ============================================================================
# Part 3: 数据清洗常量
# ============================================================================

# DataSanitizer 默认阈值
SANITIZER_PRICE_THRESHOLD: float = 3.0   # 价格异常值 MAD 阈值
SANITIZER_VOLUME_THRESHOLD: float = 5.0  # 成交量异常值 MAD 阈值
SANITIZER_MIN_VOLUME: float = 100.0      # 最小有效成交量
SANITIZER_DEFAULT_VOLUME: float = 1000000.0  # 默认成交量（中值填充用）


# ============================================================================
# Part 4: 因子计算常量
# ============================================================================

# RSRS 因子参数
RSRS_WINDOW: int = 18               # RSRS 回归窗口
RSRS_ZSCORE_WINDOW: int = 600       # Z-Score 标准化窗口
RSRS_MOMENTUM_WINDOW: int = 5       # 动量计算窗口
RSRS_R2_THRESHOLD: float = 0.8      # 有效性阈值（r² > 0.8）

# OLS 精度控制
OLS_REINIT_INTERVAL: int = 100      # 增量 OLS 重初始化间隔（避免浮点累积误差）
OLS_DENOM_THRESHOLD: float = 1e-12  # 除数阈值（避免除零）


# ============================================================================
# Part 5: 回测引擎常量
# ============================================================================

# 账户配置
DEFAULT_INITIAL_CASH: float = 1000000.0  # 默认初始资金（100 万）
DEFAULT_MAX_POSITIONS: int = 20          # 默认最大持仓数
DEFAULT_MAX_SINGLE_POSITION_RATIO: float = 0.10  # 单股最大权重（10%）
DEFAULT_MAX_TOTAL_POSITION_RATIO: float = 0.95   # 总仓位上限（95%）

# 交易成本
DEFAULT_COMMISSION_RATE: float = 0.0003     # 佣金费率（万三）
DEFAULT_MIN_COMMISSION: float = 5.0         # 最低佣金（5 元）
DEFAULT_SLIPPAGE_BASE: float = 0.0001       # 基础滑点（万一）
DEFAULT_SLIPPAGE_VOLUME_SCALE: float = 0.001   # 成交量冲击系数
DEFAULT_SLIPPAGE_VOLATILITY_SCALE: float = 0.3  # 波动率冲击系数
DEFAULT_MAX_SLIPPAGE: float = 0.05          # 最大滑点保护（5%）

# 涨跌停阈值（契约）
LIMIT_UP_THRESHOLD_NORMAL: float = 0.099    # 普通股票涨停（9.9%）
LIMIT_DOWN_THRESHOLD_NORMAL: float = 0.099  # 普通股票跌停（9.9%）
LIMIT_UP_THRESHOLD_ST: float = 0.049        # ST 股票涨停（4.9%）
LIMIT_DOWN_THRESHOLD_ST: float = 0.049      # ST 股票跌停（4.9%）
LIMIT_UP_THRESHOLD_KCB: float = 0.199       # 科创板涨停（19.9%）
LIMIT_DOWN_THRESHOLD_KCB: float = 0.199     # 科创板跌停（19.9%）


# ============================================================================
# Part 6: 性能基准常量
# ============================================================================

# 性能目标（用于测试验证）
PERFORMANCE_TARGETS: Dict[str, float] = {
    # AlphaFactorEngine 目标
    'alpha_factor_engine_1000x1250': 50.0,    # 1000 股×1250 天 < 50ms
    'alpha_factor_engine_5000x2500': 500.0,   # 5000 股×2500 天 < 500ms
    
    # TdxParallelDownloader 目标
    'collector_node_test': 5.0,               # 节点测速 < 5s
    'collector_download_speed': 15.0,         # 下载速度 > 15 stocks/s
    'collector_full_market': 360.0,           # 全市场采集 < 6 分钟
    
    # HybridExecutionEngine 目标
    'backtest_5000x250': 500.0,               # 5000 股×250 天 < 500ms
}


# ============================================================================
# Part 7: 交易日历常量
# ============================================================================

# A 股交易时间（用于分钟线）
TRADING_HOURS: Dict[str, List[Tuple[str, str]]] = {
    'stock': [
        ('09:30', '11:30'),  # 上午
        ('13:00', '15:00'),  # 下午
    ],
    'futures': [
        ('09:00', '11:30'),  # 上午
        ('13:30', '15:00'),  # 下午
        ('21:00', '23:00'),  # 夜盘
    ],
}

# 年化交易日数
ANNUAL_TRADING_DAYS: int = 252


# ============================================================================
# Part 8: 文件路径常量（契约级别）
# ============================================================================

# ✅ FIX: 目录结构（Path Hijacking 契约 - 对齐 storage.py）
# 注意：storage.py 实际使用 base_dir/parquet/daily
# 这里的常量应该与实际实现一致
DIR_PARQUET: str = "parquet"
DIR_DAILY: str = "daily"
DIR_MINUTE: str = "minute"
DIR_TICK: str = "tick"

# 完整路径模板: base_dir / parquet / daily
# （已移除 market_data 层级，对齐 storage.py 实现）
PATH_TEMPLATE_DAILY: str = f"{DIR_PARQUET}/{DIR_DAILY}"


# ============================================================================
# Part 9: 日志配置常量
# ============================================================================

# 日志级别
LOG_LEVEL_PRODUCTION: str = "INFO"
LOG_LEVEL_DEVELOPMENT: str = "DEBUG"

# 日志格式
LOG_FORMAT_SIMPLE: str = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_FORMAT_DETAILED: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# Part 10: 策略参数常量
# ============================================================================

# BaseStrategy 默认参数
STRATEGY_DEFAULT_LOOKBACK: int = 250        # 默认回看窗口
STRATEGY_DEFAULT_TOP_N: int = 50            # 默认 Top-N 选股
STRATEGY_MAX_TRADE_HISTORY: int = 10000     # 最大交易历史记录数


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 市场代码
    'Market',
    'VALID_PREFIXES_SZ',
    'VALID_PREFIXES_SH',
    
    # TDX 配置
    'TDX_BATCH_SIZE',
    'TDX_TIMEOUT',
    'TDX_MAX_FAIL_COUNT',
    'TDX_NODE_TEST_WORKERS',
    
    # 数据采集
    'COLLECTOR_MAX_WORKERS',
    'COLLECTOR_BATCH_LOG_SIZE',
    'COLLECTOR_ASYNC_LOG_QUEUE_SIZE',
    
    # 数据清洗
    'SANITIZER_PRICE_THRESHOLD',
    'SANITIZER_VOLUME_THRESHOLD',
    'SANITIZER_MIN_VOLUME',
    'SANITIZER_DEFAULT_VOLUME',
    
    # 因子计算
    'RSRS_WINDOW',
    'RSRS_ZSCORE_WINDOW',
    'RSRS_MOMENTUM_WINDOW',
    'RSRS_R2_THRESHOLD',
    'OLS_REINIT_INTERVAL',
    'OLS_DENOM_THRESHOLD',
    
    # 回测引擎
    'DEFAULT_INITIAL_CASH',
    'DEFAULT_MAX_POSITIONS',
    'DEFAULT_MAX_SINGLE_POSITION_RATIO',
    'DEFAULT_MAX_TOTAL_POSITION_RATIO',
    
    # 交易成本
    'DEFAULT_COMMISSION_RATE',
    'DEFAULT_MIN_COMMISSION',
    'DEFAULT_SLIPPAGE_BASE',
    'DEFAULT_SLIPPAGE_VOLUME_SCALE',
    'DEFAULT_SLIPPAGE_VOLATILITY_SCALE',
    'DEFAULT_MAX_SLIPPAGE',
    
    # 涨跌停阈值
    'LIMIT_UP_THRESHOLD_NORMAL',
    'LIMIT_DOWN_THRESHOLD_NORMAL',
    'LIMIT_UP_THRESHOLD_ST',
    'LIMIT_DOWN_THRESHOLD_ST',
    'LIMIT_UP_THRESHOLD_KCB',
    'LIMIT_DOWN_THRESHOLD_KCB',
    
    # 性能基准
    'PERFORMANCE_TARGETS',
    
    # 交易日历
    'TRADING_HOURS',
    'ANNUAL_TRADING_DAYS',
    
    # 文件路径
    'DIR_PARQUET',
    'DIR_DAILY',
    'PATH_TEMPLATE_DAILY',
    
    # 日志配置
    'LOG_LEVEL_PRODUCTION',
    'LOG_LEVEL_DEVELOPMENT',
    'LOG_FORMAT_SIMPLE',
    'LOG_FORMAT_DETAILED',
    'LOG_DATE_FORMAT',
    
    # 策略参数
    'STRATEGY_DEFAULT_LOOKBACK',
    'STRATEGY_DEFAULT_TOP_N',
    'STRATEGY_MAX_TRADE_HISTORY',
]
