#!/usr/bin/env python3
"""
main.py
=======
高性能量化回测系统 v2.0.1 主入口

完整数据流：
  数据采集 → 数据清洗 → 因子计算 → 策略执行 → 撮合成交 → 账户结算 → 绩效分析

契约对齐：
  - 所有路径使用 storage.parquet_dir（Path Hijacking）
  - 所有对象对齐 types.py __slots__
  - 配置从 ConfigManager 加载

使用方法：
  python main.py                    # 使用默认配置
  python main.py --config custom.yaml  # 使用自定义配置
  python main.py --download          # 仅下载数据
  python main.py --backtest          # 仅回测（跳过下载）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import ConfigManager, SystemConfig
from src.config_validator import ConfigValidator
from src.constants import *
from src.data.collector import TdxParallelDownloader
from src.data.sanitizer import DataSanitizer
from src.data.storage import ColumnarStorageManager
from src.engine.backtester import PolyStrategyBacktester
from src.engine.execution import HybridExecutionEngine
from src.strategy.strategies.rsrs_strategy import RSRSMomentumStrategy
from src.utils import ensure_path, format_number, format_percentage


# ============================================================================
# Part 1: 日志配置
# ============================================================================

def setup_logging(config: SystemConfig) -> None:
    """配置全局日志"""
    log_level = getattr(logging, config.log.level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.log.enable_file_log:
        log_file = ensure_path(config.log.log_file, create=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format=config.log.format,
        datefmt=config.log.date_format,
        handlers=handlers,
        force=True
    )
    
    logging.info("=" * 70)
    logging.info("高性能量化回测系统 v2.0.1")
    logging.info("=" * 70)


# ============================================================================
# Part 2: 数据采集流程
# ============================================================================

def run_data_collection(config: SystemConfig) -> bool:
    """
    执行数据采集流程。
    
    流程：
      1. 初始化 TdxParallelDownloader
      2. 测试节点
      3. 下载全市场数据
    
    契约：
      - 使用 storage.parquet_dir 存储
      - 使用 DataSanitizer 清洗
    """
    logging.info("\n" + "=" * 70)
    logging.info("Phase 1: 数据采集")
    logging.info("=" * 70)
    
    # 初始化组件
    storage = ColumnarStorageManager(base_dir=config.data.base_dir)
    sanitizer = DataSanitizer(
        price_threshold=config.data.sanitizer_price_threshold,
        volume_threshold=config.data.sanitizer_volume_threshold
    )
    
    downloader = TdxParallelDownloader(
        storage_manager=storage,
        data_sanitizer=sanitizer,
        max_workers=config.data.collector_max_workers,
        timeout=config.data.collector_timeout,
        enable_adjust=config.data.collector_enable_adjust,
        enable_async_log=config.data.collector_enable_async_log
    )
    
    # 测试节点
    logging.info("\n[1/2] 测试 TDX 节点...")
    available = downloader.test_nodes()
    
    if not available:
        logging.error("没有可用的 TDX 节点，数据采集失败")
        return False
    
    # 下载全市场数据
    logging.info("\n[2/2] 下载全市场数据...")
    stats = downloader.download_all_stocks()
    
    logging.info("\n✅ 数据采集完成:")
    logging.info(f"  成功: {stats['success']}")
    logging.info(f"  跳过: {stats['skip']}")
    logging.info(f"  失败: {stats['fail']}")
    logging.info(f"  总记录: {format_number(stats['total_records'])}")
    
    return stats['success'] > 0


# ============================================================================
# Part 3: 回测执行流程
# ============================================================================

def run_backtest(config: SystemConfig) -> Optional[Dict]:
    """
    执行回测流程。
    
    流程：
      1. 加载历史数据
      2. 初始化策略
      3. 执行回测
      4. 生成报告
    
    契约：
      - 策略必须继承 BaseStrategy
      - PositionState 必须使用 __slots__
      - 所有 Signal/Order/Fill 对齐 types.py
    """
    logging.info("\n" + "=" * 70)
    logging.info("Phase 2: 回测执行")
    logging.info("=" * 70)
    
    # 初始化存储
    storage = ColumnarStorageManager(base_dir=config.data.base_dir)
    
    # 加载历史数据
    logging.info("\n[1/4] 加载历史数据...")
    
    try:
        # 获取所有股票代码
        parquet_files = list(storage.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            logging.error(f"未找到数据文件: {storage.parquet_dir}")
            logging.info("提示: 请先运行数据采集 (python main.py --download)")
            return None
        
        logging.info(f"发现 {len(parquet_files)} 个数据文件")
        
        # 限制股票数量（避免内存溢出）
        max_stocks = 100  # 可调整
        codes = [f.stem for f in parquet_files[:max_stocks]]
        
        # 加载数据
        history: Dict[str, pd.DataFrame] = {}
        
        for code in codes:
            df = storage.load_stock_data(code)
            if df is None or df.empty:
                continue
            
            # ✅ FIX: 统一使用date列过滤
            # 检查数据结构
            if "date" not in df.columns:
                # 如果没有date列，尝试从index转换
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    df = df.rename(columns={"index": "date"})
                else:
                    logging.warning(f"{code} 缺少date列，跳过")
                    continue
            
            # 确保date是字符串格式
            if not pd.api.types.is_string_dtype(df["date"]):
                try:
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                except Exception as e:
                    logging.warning(f"{code} 日期转换失败: {e}")
                    continue
            
            # 过滤日期范围
            df_filtered = df[
                (df["date"] >= config.backtest.start_date) &
                (df["date"] <= config.backtest.end_date)
            ].copy()
            
            if not df_filtered.empty:
                # 设置index为date（execution.py需要）
                df_filtered = df_filtered.set_index("date")
                history[code] = df_filtered
        
        if not history:
            logging.error("加载的历史数据为空")
            return None
        
        logging.info(f"✓ 成功加载 {len(history)} 只股票数据")
        
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        return None
    
    # 初始化策略
    logging.info("\n[2/4] 初始化策略...")
    
    strategy = RSRSMomentumStrategy(
        top_n=50,
        lookback=250,
        rsrs_threshold=0.5,
        stop_loss_pct=-0.05,
        enable_stop_loss=True
    )
    
    logging.info(f"✓ 策略: {strategy.name}")
    
    # 创建回测引擎
    logging.info("\n[3/4] 创建回测引擎...")
    
    engine = HybridExecutionEngine(
        strategy=strategy,
        initial_cash=config.account.initial_cash,
        start_date=config.backtest.start_date,
        end_date=config.backtest.end_date,
        commission_rate=config.trading_cost.commission_rate,
    )
    
    logging.info("✓ 引擎初始化完成")
    
    # 执行回测
    logging.info("\n[4/4] 执行回测...")
    logging.info(f"  起始日期: {config.backtest.start_date}")
    logging.info(f"  结束日期: {config.backtest.end_date}")
    logging.info(f"  初始资金: {format_number(config.account.initial_cash)}")
    
    try:
        start_time = time.time()
        
        result = engine.run_backtest(history)
        
        elapsed = time.time() - start_time
        
        logging.info(f"✅ 回测完成 (耗时 {elapsed:.1f}s)")
        
        return result
    
    except Exception as e:
        logging.error(f"回测执行失败: {e}", exc_info=True)
        return None


# ============================================================================
# Part 4: 结果分析与报告
# ============================================================================

def generate_report(result: Dict, config: SystemConfig) -> None:
    """生成回测报告"""
    logging.info("\n" + "=" * 70)
    logging.info("Phase 3: 绩效分析")
    logging.info("=" * 70)
    
    # 提取关键指标
    snapshots = result.get('snapshots', [])
    if not snapshots:
        logging.warning("无快照数据")
        return
    
    # 计算关键指标
    initial_equity = snapshots[0].total_value
    final_equity = snapshots[-1].total_value
    total_return = (final_equity - initial_equity) / initial_equity
    
    logging.info("\n📊 回测摘要:")
    logging.info(f"  初始资金: {format_number(initial_equity)}")
    logging.info(f"  最终资金: {format_number(final_equity)}")
    logging.info(f"  总收益率: {format_percentage(total_return)}")
    
    # 计算最大回撤
    equity_curve = [s.total_value for s in snapshots]
    cummax = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - cummax) / cummax
    max_dd = drawdown.min()
    
    logging.info(f"  最大回撤: {format_percentage(max_dd)}")
    
    # 统计信息
    stats = result.get('stats', {})
    logging.info(f"\n📈 交易统计:")
    logging.info(f"  总交易次数: {stats.get('total_trades', 0)}")
    logging.info(f"  盈利次数: {stats.get('winning_trades', 0)}")
    logging.info(f"  亏损次数: {stats.get('losing_trades', 0)}")
    
    win_rate = 0.0
    if stats.get('total_trades', 0) > 0:
        win_rate = stats.get('winning_trades', 0) / stats['total_trades']
    logging.info(f"  胜率: {format_percentage(win_rate)}")
    
    logging.info("\n" + "=" * 70)


# ============================================================================
# Part 5: 主函数
# ============================================================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="高性能量化回测系统 v2.0.1"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='仅执行数据下载'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='仅执行回测（跳过数据下载）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = ConfigManager.load(args.config)
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        print("使用默认配置继续...")
        config = ConfigManager.load(None)
    
    # 配置日志
    setup_logging(config)
    
    # ✅ 验证配置
    try:
        ConfigValidator.validate_all(config)
    except ValueError as e:
        logging.error(f"配置验证失败: {e}")
        sys.exit(1)
    
    # 执行流程
    success = True
    
    if args.download:
        # 仅下载数据
        success = run_data_collection(config)
    
    elif args.backtest:
        # 仅回测
        result = run_backtest(config)
        if result:
            generate_report(result, config)
        else:
            success = False
    
    else:
        # 完整流程
        if config.data.enable_collector:
            if not run_data_collection(config):
                logging.error("数据采集失败，跳过回测")
                success = False
        
        if success:
            result = run_backtest(config)
            if result:
                generate_report(result, config)
            else:
                success = False
    
    # 退出
    if success:
        logging.info("\n✅ 系统执行成功")
        sys.exit(0)
    else:
        logging.error("\n❌ 系统执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
