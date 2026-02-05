"""
src/utils.py
============
系统通用工具函数（v2.0.1 标准）

职责：
  1. 日期时间处理
  2. 路径处理（六边形架构适配器）
  3. 数据验证
  4. 性能监控装饰器
"""

from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from .constants import ANNUAL_TRADING_DAYS


# ============================================================================
# Part 1: 日期时间工具
# ============================================================================

def parse_date(date_str: Union[str, datetime, pd.Timestamp]) -> datetime:
    """
    解析日期字符串为 datetime。
    
    支持格式：
      - "2024-01-01"
      - "20240101"
      - datetime 对象
      - pd.Timestamp 对象
    """
    if isinstance(date_str, datetime):
        return date_str
    
    if isinstance(date_str, pd.Timestamp):
        return date_str.to_pydatetime()
    
    if isinstance(date_str, str):
        # 尝试解析常见格式
        for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"无法解析日期格式: {date_str}")
    
    raise TypeError(f"不支持的日期类型: {type(date_str)}")


def format_date(dt: Union[datetime, pd.Timestamp], fmt: str = "%Y-%m-%d") -> str:
    """格式化日期为字符串"""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    
    if isinstance(dt, datetime):
        return dt.strftime(fmt)
    
    raise TypeError(f"不支持的日期类型: {type(dt)}")


def get_trading_dates(
    start_date: str,
    end_date: str,
    calendar: Optional[List[str]] = None
) -> List[str]:
    """
    获取交易日列表（简化版）。
    
    Args:
        start_date: 起始日期
        end_date: 结束日期
        calendar: 交易日历（None 则使用工作日近似）
    
    Returns:
        交易日列表 ["2024-01-01", ...]
    
    注意：
        生产环境建议使用 pandas_market_calendars
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    if calendar:
        # 使用提供的日历
        return [
            d for d in calendar
            if start <= parse_date(d) <= end
        ]
    
    # 简化版：使用工作日近似（排除周末）
    dates = pd.date_range(start, end, freq='B')  # B = business day
    return [d.strftime("%Y-%m-%d") for d in dates]


# ============================================================================
# Part 2: 路径处理工具（六边形架构适配器）
# ============================================================================

def ensure_path(path: Union[str, Path], create: bool = True) -> Path:
    """
    确保路径存在（六边形架构适配器）。
    
    Args:
        path: 路径
        create: 是否创建不存在的路径
    
    Returns:
        Path 对象
    
    契约：
        - 所有路径处理必须使用此函数
        - 避免业务代码直接操作文件系统
    """
    p = Path(path)
    
    if create and not p.exists():
        if '.' in p.name:  # 文件
            p.parent.mkdir(parents=True, exist_ok=True)
        else:  # 目录
            p.mkdir(parents=True, exist_ok=True)
    
    return p


def safe_path_join(*parts: Union[str, Path]) -> Path:
    """
    安全路径拼接（防止路径穿越攻击）。
    
    契约：
        - 禁止使用 '..'
        - 禁止使用绝对路径拼接
    """
    result = Path(parts[0])
    
    for part in parts[1:]:
        part_str = str(part)
        
        # 安全检查
        if '..' in part_str:
            raise ValueError(f"路径包含非法字符 '..': {part_str}")
        
        if Path(part_str).is_absolute():
            raise ValueError(f"不允许拼接绝对路径: {part_str}")
        
        result = result / part_str
    
    return result


# ============================================================================
# Part 3: 数据验证工具
# ============================================================================

def validate_ohlcv(df: pd.DataFrame, strict: bool = True) -> bool:
    """
    验证 OHLCV 数据完整性。
    
    Args:
        df: DataFrame（必须包含 open/high/low/close/volume）
        strict: 严格模式（检查数据合理性）
    
    Returns:
        是否有效
    
    契约：
        - 必须包含 open/high/low/close/volume
        - 价格 > 0
        - high >= max(open, close)
        - low <= min(open, close)
        - volume >= 0
    """
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        logging.warning(f"缺少必需字段: {missing}")
        return False
    
    if df.empty:
        logging.warning("DataFrame 为空")
        return False
    
    if not strict:
        return True
    
    # 严格检查
    try:
        # 价格 > 0
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                logging.warning(f"{col} 包含 <= 0 的值")
                return False
        
        # high >= max(open, close)
        if (df['high'] < df[['open', 'close']].max(axis=1)).any():
            logging.warning("high < max(open, close)")
            return False
        
        # low <= min(open, close)
        if (df['low'] > df[['open', 'close']].min(axis=1)).any():
            logging.warning("low > min(open, close)")
            return False
        
        # volume >= 0
        if (df['volume'] < 0).any():
            logging.warning("volume < 0")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"数据验证失败: {e}")
        return False


def check_data_alignment(
    dfs: Dict[str, pd.DataFrame],
    date_col: Optional[str] = None
) -> bool:
    """
    检查多个 DataFrame 的日期对齐。
    
    Args:
        dfs: {code: DataFrame}
        date_col: 日期列名（None 则使用 index）
    
    Returns:
        是否对齐
    """
    if not dfs:
        return True
    
    date_sets = []
    
    for code, df in dfs.items():
        if date_col:
            if date_col not in df.columns:
                logging.warning(f"{code} 缺少日期列 {date_col}")
                return False
            dates = set(df[date_col].astype(str))
        else:
            dates = set(df.index.astype(str))
        
        date_sets.append(dates)
    
    # 检查是否有公共日期
    common_dates = set.intersection(*date_sets)
    
    if not common_dates:
        logging.warning("无公共交易日")
        return False
    
    # 检查是否完全一致
    if len(set(len(s) for s in date_sets)) > 1:
        logging.info(
            f"日期数量不一致: {[len(s) for s in date_sets[:5]]}..."
        )
    
    return True


# ============================================================================
# Part 4: 性能监控装饰器
# ============================================================================

F = TypeVar('F', bound=Callable[..., Any])


def timer(func: F) -> F:
    """
    函数计时装饰器。
    
    用法：
        @timer
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        logging.debug(f"{func.__name__} 耗时: {elapsed*1000:.2f}ms")
        
        return result
    
    return wrapper  # type: ignore


def benchmark(iterations: int = 1):
    """
    性能基准测试装饰器。
    
    用法：
        @benchmark(iterations=100)
        def my_function():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            result = None
            
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            avg_time = np.mean(times) * 1000
            min_time = np.min(times) * 1000
            max_time = np.max(times) * 1000
            
            logging.info(
                f"{func.__name__} 基准测试 ({iterations} 次): "
                f"平均 {avg_time:.2f}ms | 最小 {min_time:.2f}ms | 最大 {max_time:.2f}ms"
            )
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


# ============================================================================
# Part 5: 数值计算工具
# ============================================================================

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.03,
    annual_factor: int = ANNUAL_TRADING_DAYS
) -> float:
    """
    计算夏普比率。
    
    Args:
        returns: 日收益率序列
        risk_free_rate: 无风险利率（年化）
        annual_factor: 年化因子（默认 252）
    
    Returns:
        夏普比率
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    daily_rf = risk_free_rate / annual_factor
    sharpe = (mean_return - daily_rf) / std_return * np.sqrt(annual_factor)
    
    return sharpe


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    计算最大回撤。
    
    Args:
        equity_curve: 权益曲线
    
    Returns:
        最大回撤（负数）
    """
    if len(equity_curve) == 0:
        return 0.0
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    
    return np.min(drawdown)


def calculate_calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    annual_factor: int = ANNUAL_TRADING_DAYS
) -> float:
    """
    计算卡玛比率（年化收益率 / 最大回撤）。
    
    Args:
        returns: 日收益率序列
        equity_curve: 权益曲线
        annual_factor: 年化因子
    
    Returns:
        卡玛比率
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
    
    annual_return = np.mean(returns) * annual_factor
    max_dd = abs(calculate_max_drawdown(equity_curve))
    
    if max_dd == 0:
        return 0.0
    
    return annual_return / max_dd


# ============================================================================
# Part 6: 字符串处理工具
# ============================================================================

def truncate_string(s: str, max_length: int = 50, suffix: str = "...") -> str:
    """截断字符串"""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def format_number(
    num: float,
    precision: int = 2,
    use_separator: bool = True
) -> str:
    """格式化数字"""
    if use_separator:
        return f"{num:,.{precision}f}"
    return f"{num:.{precision}f}"


def format_percentage(num: float, precision: int = 2) -> str:
    """格式化百分比"""
    return f"{num * 100:.{precision}f}%"


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 日期时间
    'parse_date',
    'format_date',
    'get_trading_dates',
    
    # 路径处理
    'ensure_path',
    'safe_path_join',
    
    # 数据验证
    'validate_ohlcv',
    'check_data_alignment',
    
    # 性能监控
    'timer',
    'benchmark',
    
    # 数值计算
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    
    # 字符串处理
    'truncate_string',
    'format_number',
    'format_percentage',
]
