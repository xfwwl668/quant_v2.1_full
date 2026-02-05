#!/usr/bin/env python3
"""
run_backtest.py
===============
快速回测启动脚本（开发/调试用）

特点：
  - 使用小规模数据集（10 只股票）
  - 短时间窗口（3 个月）
  - 详细日志输出
  - 快速验证策略逻辑

使用方法：
  python run_backtest.py           # 使用默认参数
  python run_backtest.py --codes 000001 000002  # 指定股票
  python run_backtest.py --start 2024-01-01 --end 2024-03-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.storage import ColumnarStorageManager
from src.engine.execution import HybridExecutionEngine
from src.strategy.strategies.rsrs_strategy import RSRSMomentumStrategy


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )


def load_test_data(
    base_dir: str = "./data",
    codes: Optional[List[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31"
) -> Dict[str, pd.DataFrame]:
    """加载测试数据"""
    storage = ColumnarStorageManager(base_dir=base_dir)
    
    # 使用指定股票或默认股票
    if codes is None:
        codes = [
            '000001', '000002', '000333', '000651',
            '600000', '600036', '600519', '600887',
            '688001', '688008'
        ]
    
    logging.info(f"加载测试数据: {len(codes)} 只股票")
    
    history: Dict[str, pd.DataFrame] = {}
    
    for code in codes:
        df = storage.load_stock_data(code)
        
        if df is not None and not df.empty:
            # 过滤日期
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if not df.empty:
                history[code] = df
                logging.debug(f"  ✓ {code}: {len(df)} 条记录")
    
    if not history:
        logging.error("未加载到任何数据")
        sys.exit(1)
    
    logging.info(f"✓ 成功加载 {len(history)} 只股票")
    
    return history


def run_quick_backtest(
    history: Dict[str, pd.DataFrame],
    initial_cash: float = 1000000.0,
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31"
):
    """执行快速回测"""
    logging.info("\n" + "=" * 60)
    logging.info("快速回测启动")
    logging.info("=" * 60)
    
    # 创建策略
    strategy = RSRSMomentumStrategy(
        top_n=5,           # 少量持仓
        lookback=60,       # 短回看窗口
        rsrs_threshold=0.3,
        stop_loss_pct=-0.05
    )
    
    logging.info(f"\n策略配置:")
    logging.info(f"  名称: {strategy.name}")
    logging.info(f"  Top-N: {strategy.top_n}")
    logging.info(f"  回看窗口: {strategy.lookback}")
    
    # 创建引擎
    engine = HybridExecutionEngine(
        strategy=strategy,
        initial_cash=initial_cash,
        start_date=start_date,
        end_date=end_date,
        commission_rate=0.0003,
    )
    
    # 执行回测
    logging.info("\n执行回测...")
    
    try:
        result = engine.run_backtest(history)
        
        # 输出结果
        snapshots = result.get('snapshots', [])
        
        if snapshots:
            logging.info(f"\n{'='*60}")
            logging.info("回测结果摘要")
            logging.info(f"{'='*60}")
            
            initial = snapshots[0].total_value
            final = snapshots[-1].total_value
            ret = (final - initial) / initial
            
            logging.info(f"初始资金: {initial:,.2f}")
            logging.info(f"最终资金: {final:,.2f}")
            logging.info(f"总收益率: {ret*100:.2f}%")
            
            # 最大回撤
            equity = [s.total_value for s in snapshots]
            cummax = pd.Series(equity).cummax()
            dd = (pd.Series(equity) - cummax) / cummax
            max_dd = dd.min()
            
            logging.info(f"最大回撤: {max_dd*100:.2f}%")
            
            # 交易统计
            stats = result.get('stats', {})
            logging.info(f"\n交易次数: {stats.get('total_trades', 0)}")
            logging.info(f"胜率: {stats.get('winning_trades', 0) / max(stats.get('total_trades', 1), 1)*100:.1f}%")
            
            logging.info(f"\n{'='*60}")
        
        else:
            logging.warning("无快照数据")
    
    except Exception as e:
        logging.error(f"回测失败: {e}", exc_info=True)
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速回测脚本")
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='./data',
        help='数据目录'
    )
    parser.add_argument(
        '--codes',
        nargs='+',
        help='股票代码列表'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2024-01-01',
        help='起始日期'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2024-03-31',
        help='结束日期'
    )
    parser.add_argument(
        '--cash',
        type=float,
        default=1000000.0,
        help='初始资金'
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging()
    
    # 加载数据
    history = load_test_data(
        base_dir=args.base_dir,
        codes=args.codes,
        start_date=args.start,
        end_date=args.end
    )
    
    # 执行回测
    run_quick_backtest(
        history=history,
        initial_cash=args.cash,
        start_date=args.start,
        end_date=args.end
    )
    
    logging.info("\n✅ 快速回测完成")


if __name__ == "__main__":
    main()
