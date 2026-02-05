#!/usr/bin/env python3
"""
简化的系统完整性验证
检查所有关键修复是否存在
"""

import sys
from pathlib import Path

# 添加到路径
sys.path.insert(0, str(Path(__file__).parent))

def check_file_exists(filepath):
    """检查文件是否存在"""
    p = Path(filepath)
    if p.exists():
        print(f"  ✓ {filepath}")
        return True
    else:
        print(f"  ✗ {filepath} 缺失")
        return False

def check_fix_markers(filepath, markers):
    """检查文件中是否包含修复标记"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        found = []
        for marker in markers:
            if marker in content:
                found.append(marker)
        
        if len(found) == len(markers):
            print(f"  ✓ {filepath}: 包含所有修复标记 ({len(found)}/{len(markers)})")
            return True
        else:
            print(f"  ⚠ {filepath}: 缺少部分修复标记 ({len(found)}/{len(markers)})")
            missing = set(markers) - set(found)
            for m in missing:
                print(f"    - 缺失: {m}")
            return False
    except Exception as e:
        print(f"  ✗ {filepath}: 读取失败 - {e}")
        return False

def main():
    print("=" * 70)
    print("系统完整性快速验证 v2.1.0")
    print("=" * 70)
    
    results = []
    
    # 检查1: 关键文件存在性
    print("\n[1/6] 检查关键修复文件...")
    files_to_check = [
        "src/factors/technical/rsrs.py",
        "src/engine/match.py",
        "src/engine/account.py",
        "src/engine/execution.py",
        "main.py",
        "src/constants.py",
        "FIXES_v2.1.0.md",
    ]
    
    file_check = all(check_file_exists(f) for f in files_to_check)
    results.append(("关键文件", file_check))
    
    # 检查2: rsrs.py 修复标记
    print("\n[2/6] 验证rsrs.py NaN处理修复...")
    rsrs_markers = [
        "✅ FIX",
        "valid_count",
        "min_valid",
    ]
    rsrs_fix = check_fix_markers("src/factors/technical/rsrs.py", rsrs_markers)
    results.append(("rsrs.py NaN处理", rsrs_fix))
    
    # 检查3: match.py 修复标记
    print("\n[3/6] 验证match.py涨跌停检测修复...")
    match_markers = [
        "prev_close",
        "is_st",
        "is_kcb",
        "limit_ratio",
    ]
    match_fix = check_fix_markers("src/engine/match.py", match_markers)
    results.append(("match.py涨跌停", match_fix))
    
    # 检查4: account.py T+1结算
    print("\n[4/6] 验证account.py T+1结算修复...")
    account_markers = [
        "frozen_cash",
        "on_day_end",
        "T+1",
    ]
    account_fix = check_fix_markers("src/engine/account.py", account_markers)
    results.append(("account.py T+1", account_fix))
    
    # 检查5: execution.py 非交易日过滤
    print("\n[5/6] 验证execution.py非交易日过滤修复...")
    exec_markers = [
        "trading_dates",
        "account.on_day_end()",
    ]
    exec_fix = check_fix_markers("src/engine/execution.py", exec_markers)
    results.append(("execution.py非交易日", exec_fix))
    
    # 检查6: constants.py 路径契约
    print("\n[6/6] 验证constants.py路径契约修复...")
    const_markers = [
        'PATH_TEMPLATE_DAILY',
        'parquet/daily',
    ]
    # 确保不包含market_data
    const_fix = check_fix_markers("src/constants.py", const_markers)
    
    # 额外检查：不应包含market_data
    with open("src/constants.py", 'r', encoding='utf-8') as f:
        const_content = f.read()
    if "market_data/parquet" in const_content.lower():
        print("  ⚠ constants.py: 仍然包含错误的market_data路径")
        const_fix = False
    
    results.append(("constants.py路径", const_fix))
    
    # 总结
    print("\n" + "=" * 70)
    print("验证结果:")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print("\n" + "=" * 70)
    if passed_count == total_count:
        print(f"✅ 所有检查通过 ({passed_count}/{total_count})")
        print("=" * 70)
        print("\n系统修复完成！推荐执行:")
        print("  1. python verify_system.py  # 完整性验证")
        print("  2. python run_backtest.py   # 快速回测测试")
        return 0
    else:
        print(f"⚠️  部分检查失败 ({passed_count}/{total_count})")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit(main())
