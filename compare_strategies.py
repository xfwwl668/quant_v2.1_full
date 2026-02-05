#!/usr/bin/env python3
"""
多策略对比测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import logging

from src.config import ConfigManager
from src.engine.execution import HybridExecutionEngine
from src.data.storage import ColumnarStorageManager
from src.strategy.strategies import (
    RSRSMomentumStrategy,
    RSRSAdvancedStrategy,
    AlphaHunterStrategy,
    ShortTermStrategy,
    MomentumReversalStrategy,
    SentimentReversalStrategy,
)

logging.basicConfig(level=logging.WARNING)

def load_data(max_stocks=30):
    """加载数据"""
    storage = ColumnarStorageManager(base_dir="./data")
    parquet_files = list(storage.parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        return None
    
    codes = [f.stem for f in parquet_files[:max_stocks]]
    history = {}
    
    for code in codes:
        df = storage.load_stock_data(code)
        if df is not None and not df.empty:
            df_filtered = df[
                (df.index >= "2024-01-01") &
                (df.index <= "2024-06-30")
            ]
            if not df_filtered.empty:
                history[code] = df_filtered
    
    return history

def test_strategy(strategy, history, name):
    """测试策略"""
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    
    engine = HybridExecutionEngine(
        strategy=strategy,
        initial_cash=1000000.0,
        start_date="2024-01-01",
        end_date="2024-06-30",
    )
    
    try:
        result = engine.run_backtest(history)
        snapshots = result.get('snapshots', [])
        
        if snapshots:
            initial = snapshots[0].total_value
            final = snapshots[-1].total_value
            ret = (final - initial) / initial
            
            equity = [s.total_value for s in snapshots]
            cummax = pd.Series(equity).cummax()
            dd = (pd.Series(equity) - cummax) / cummax
            max_dd = dd.min()
            
            trade_stats = result.get('trade_statistics', {})
            total_trades = len(result.get('trade_history', []))
            
            print(f"收益率: {ret*100:.2f}%")
            print(f"最大回撤: {max_dd*100:.2f}%")
            print(f"交易次数: {total_trades}")
            
            return {'name': name, 'return': ret, 'max_dd': max_dd, 'trades': total_trades}
    except Exception as e:
        print(f"❌ 失败: {e}")
        return None

def main():
    print("="*60)
    print("多策略对比测试")
    print("="*60)
    
    history = load_data(30)
    if not history:
        print("❌ 数据加载失败")
        return
    
    print(f"✓ 加载 {len(history)} 只股票")
    
    strategies = [
        (RSRSMomentumStrategy(top_n=20), "RSRS动量"),
        (RSRSAdvancedStrategy(top_n=18), "RSRS高级"),
        (AlphaHunterStrategy(top_n=15), "Alpha Hunter"),
        (ShortTermStrategy(top_n=12), "短线"),
        (MomentumReversalStrategy(), "动量反转"),
        (SentimentReversalStrategy(top_n=10), "情绪反转"),
    ]
    
    results = []
    for strategy, name in strategies:
        result = test_strategy(strategy, history, name)
        if result:
            results.append(result)
    
    if results:
        print("\n" + "="*60)
        print("汇总")
        print("="*60)
        print(f"{'策略':<16} {'收益率':>10} {'回撤':>10} {'交易':>8}")
        print("-"*60)
        
        for r in results:
            print(f"{r['name']:<16} {r['return']*100:>9.2f}% {r['max_dd']*100:>9.2f}% {r['trades']:>8}")
        
        best = max(results, key=lambda x: x['return'])
        print(f"\n🏆 最高收益: {best['name']} ({best['return']*100:.2f}%)")

if __name__ == "__main__":
    main()
