"""
src/config.py
=============
配置管理器（v2.0.1 架构标准）

职责：
  1. 加载配置文件（config.yaml）
  2. 提供类型安全的配置访问
  3. 支持环境变量覆盖
  4. 验证配置完整性

契约：
  - 所有配置必须有默认值（from constants.py）
  - 配置修改必须类型安全
  - 支持热重载（开发模式）
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .constants import *


# ============================================================================
# Part 1: 配置数据类
# ============================================================================

@dataclass
class DataConfig:
    """数据相关配置"""
    base_dir: str = "./data"
    enable_collector: bool = True
    enable_sanitizer: bool = True
    
    # 采集配置
    collector_max_workers: int = COLLECTOR_MAX_WORKERS
    collector_timeout: float = TDX_TIMEOUT
    collector_enable_adjust: bool = True
    collector_enable_async_log: bool = True
    
    # 清洗配置
    sanitizer_price_threshold: float = SANITIZER_PRICE_THRESHOLD
    sanitizer_volume_threshold: float = SANITIZER_VOLUME_THRESHOLD


@dataclass
class FactorConfig:
    """因子计算配置"""
    # RSRS 参数
    rsrs_window: int = RSRS_WINDOW
    rsrs_zscore_window: int = RSRS_ZSCORE_WINDOW
    rsrs_momentum_window: int = RSRS_MOMENTUM_WINDOW
    rsrs_r2_threshold: float = RSRS_R2_THRESHOLD


@dataclass
class AccountConfig:
    """账户配置"""
    initial_cash: float = DEFAULT_INITIAL_CASH
    max_positions: int = DEFAULT_MAX_POSITIONS
    max_single_position_ratio: float = DEFAULT_MAX_SINGLE_POSITION_RATIO
    max_total_position_ratio: float = DEFAULT_MAX_TOTAL_POSITION_RATIO


@dataclass
class TradingCostConfig:
    """交易成本配置"""
    commission_rate: float = DEFAULT_COMMISSION_RATE
    min_commission: float = DEFAULT_MIN_COMMISSION
    slippage_base: float = DEFAULT_SLIPPAGE_BASE
    slippage_volume_scale: float = DEFAULT_SLIPPAGE_VOLUME_SCALE
    slippage_volatility_scale: float = DEFAULT_SLIPPAGE_VOLATILITY_SCALE
    max_slippage: float = DEFAULT_MAX_SLIPPAGE


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    benchmark: str = "000300"  # 沪深 300
    show_progress: bool = True


@dataclass
class LogConfig:
    """日志配置"""
    level: str = LOG_LEVEL_PRODUCTION
    format: str = LOG_FORMAT_SIMPLE
    date_format: str = LOG_DATE_FORMAT
    enable_file_log: bool = True
    log_file: str = "backtest.log"


@dataclass
class SystemConfig:
    """系统总配置"""
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    trading_cost: TradingCostConfig = field(default_factory=TradingCostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    # 系统级配置
    environment: str = "production"  # production / development
    enable_performance_monitoring: bool = False
    
    def __post_init__(self):
        """配置验证"""
        # 验证日期格式
        try:
            from datetime import datetime
            datetime.strptime(self.backtest.start_date, "%Y-%m-%d")
            datetime.strptime(self.backtest.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"日期格式错误: {e}")
        
        # 验证资金
        if self.account.initial_cash <= 0:
            raise ValueError("initial_cash 必须 > 0")
        
        # 验证仓位比例
        if not (0 < self.account.max_single_position_ratio <= 1.0):
            raise ValueError("max_single_position_ratio 必须在 (0, 1] 范围内")
        
        if not (0 < self.account.max_total_position_ratio <= 1.0):
            raise ValueError("max_total_position_ratio 必须在 (0, 1] 范围内")


# ============================================================================
# Part 2: 配置管理器
# ============================================================================

class ConfigManager:
    """
    配置管理器（单例模式）。
    
    使用方法：
        config = ConfigManager.load("config.yaml")
        print(config.account.initial_cash)
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[SystemConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(
        cls,
        config_file: Optional[str] = None,
        env_prefix: str = "QUANT_"
    ) -> SystemConfig:
        """
        加载配置文件。
        
        Args:
            config_file: 配置文件路径（None 则使用默认配置）
            env_prefix: 环境变量前缀（用于覆盖配置）
        
        Returns:
            SystemConfig 实例
        """
        instance = cls()
        
        # Step 1: 加载默认配置
        config_dict: Dict[str, Any] = {}
        
        # Step 2: 从文件加载（如果存在）
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config_dict = yaml_config
                logging.info(f"✓ 配置文件加载成功: {config_file}")
            except Exception as e:
                logging.warning(f"配置文件加载失败: {e}，使用默认配置")
        
        # Step 3: 环境变量覆盖
        env_overrides = cls._load_from_env(env_prefix)
        cls._deep_update(config_dict, env_overrides)
        
        # Step 4: 构建配置对象
        try:
            config = SystemConfig(
                data=DataConfig(**config_dict.get('data', {})),
                factor=FactorConfig(**config_dict.get('factor', {})),
                account=AccountConfig(**config_dict.get('account', {})),
                trading_cost=TradingCostConfig(**config_dict.get('trading_cost', {})),
                backtest=BacktestConfig(**config_dict.get('backtest', {})),
                log=LogConfig(**config_dict.get('log', {})),
                environment=config_dict.get('environment', 'production'),
                enable_performance_monitoring=config_dict.get('enable_performance_monitoring', False),
            )
        except Exception as e:
            logging.error(f"配置解析失败: {e}")
            raise
        
        instance._config = config
        return config
    
    @staticmethod
    def _load_from_env(prefix: str) -> Dict[str, Any]:
        """从环境变量加载配置覆盖"""
        overrides: Dict[str, Any] = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # QUANT_ACCOUNT_INITIAL_CASH -> account.initial_cash
                config_key = key[len(prefix):].lower().replace('_', '.')
                parts = config_key.split('.')
                
                # 简单类型转换
                if value.lower() in ('true', 'false'):
                    parsed_value: Any = value.lower() == 'true'
                elif value.replace('.', '', 1).isdigit():
                    parsed_value = float(value) if '.' in value else int(value)
                else:
                    parsed_value = value
                
                # 构建嵌套字典
                current = overrides
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = parsed_value
        
        return overrides
    
    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> None:
        """深度更新字典"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                ConfigManager._deep_update(base[key], value)
            else:
                base[key] = value
    
    @classmethod
    def get_config(cls) -> SystemConfig:
        """获取当前配置（必须先 load）"""
        instance = cls()
        if instance._config is None:
            raise RuntimeError("配置未加载，请先调用 ConfigManager.load()")
        return instance._config
    
    @classmethod
    def reload(cls, config_file: Optional[str] = None) -> SystemConfig:
        """重新加载配置（热重载）"""
        logging.info("重新加载配置...")
        return cls.load(config_file)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'SystemConfig',
    'DataConfig',
    'FactorConfig',
    'AccountConfig',
    'TradingCostConfig',
    'BacktestConfig',
    'LogConfig',
    'ConfigManager',
]


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例 1: 使用默认配置
    config = ConfigManager.load()
    print(f"初始资金: {config.account.initial_cash}")
    
    # 示例 2: 从文件加载
    # config = ConfigManager.load("config.yaml")
    
    # 示例 3: 环境变量覆盖
    # export QUANT_ACCOUNT_INITIAL_CASH=2000000
    # config = ConfigManager.load()
    
    # 示例 4: 获取已加载的配置
    # config = ConfigManager.get_config()
