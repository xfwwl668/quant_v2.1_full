"""
配置验证器
确保配置文件参数合法
"""

from typing import Dict, List, Any
import logging


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_account(config: Dict[str, Any]) -> List[str]:
        """验证账户配置"""
        errors = []
        
        initial_cash = config.get('initial_cash', 0)
        if initial_cash <= 0:
            errors.append(f"initial_cash必须>0，当前: {initial_cash}")
        
        max_positions = config.get('max_positions', 0)
        if max_positions <= 0:
            errors.append(f"max_positions必须>0，当前: {max_positions}")
        
        return errors
    
    @staticmethod
    def validate_trading_cost(config: Dict[str, Any]) -> List[str]:
        """验证交易成本配置"""
        errors = []
        
        commission_rate = config.get('commission_rate', -1)
        if not 0 <= commission_rate <= 0.01:
            errors.append(f"commission_rate必须在[0, 0.01]，当前: {commission_rate}")
        
        return errors
    
    @staticmethod
    def validate_backtest(config: Dict[str, Any]) -> List[str]:
        """验证回测配置"""
        errors = []
        
        start_date = config.get('start_date', '')
        end_date = config.get('end_date', '')
        
        if not start_date:
            errors.append("start_date不能为空")
        if not end_date:
            errors.append("end_date不能为空")
        
        if start_date and end_date and start_date >= end_date:
            errors.append(f"start_date必须<end_date: {start_date} >= {end_date}")
        
        return errors
    
    @classmethod
    def validate_all(cls, config: Any) -> None:
        """验证所有配置"""
        errors = []
        
        if hasattr(config, 'account'):
            errors.extend(cls.validate_account(config.account.__dict__))
        
        if hasattr(config, 'trading_cost'):
            errors.extend(cls.validate_trading_cost(config.trading_cost.__dict__))
        
        if hasattr(config, 'backtest'):
            errors.extend(cls.validate_backtest(config.backtest.__dict__))
        
        if errors:
            error_msg = "\n配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors)
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info("✓ 配置验证通过")
