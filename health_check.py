#!/usr/bin/env python3
"""
系统健康检查
检查所有关键问题是否已修复
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """检查策略导入"""
    print("[1/5] 检查策略导入...")
    try:
        from src.strategy.strategies import (
            RSRSMomentumStrategy,
            RSRSAdvancedStrategy,
            AlphaHunterStrategy,
            ShortTermStrategy,
            MomentumReversalStrategy,
            SentimentReversalStrategy,
        )
        print("  ✓ 6个策略导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return False

def check_bare_except():
    """检查裸except"""
    print("\n[2/5] 检查裸except...")
    issues = []
    
    for file in ['src/data/collector.py', 'src/strategy/strategies/short_term.py']:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'except:' in line and 'except Exception' not in line:
                    issues.append(f"{file}:{i}")
        except Exception as e:
            issues.append(f"{file}: 读取失败 - {e}")
    
    if issues:
        print(f"  ✗ 发现{len(issues)}处裸except:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✓ 无裸except")
        return True

def check_config_validator():
    """检查配置验证"""
    print("\n[3/5] 检查配置验证...")
    try:
        from src.config_validator import ConfigValidator
        print("  ✓ ConfigValidator存在")
        
        # 检查main.py是否使用
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'ConfigValidator' in content:
            print("  ✓ main.py已集成ConfigValidator")
            return True
        else:
            print("  ✗ main.py未使用ConfigValidator")
            return False
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

def check_execution_params():
    """检查HybridExecutionEngine参数"""
    print("\n[4/5] 检查引擎参数...")
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'slippage_base=' in content:
            print("  ✗ main.py仍包含slippage_base参数")
            return False
        else:
            print("  ✓ slippage_base参数已移除")
            return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

def check_thread_safety():
    """检查并发安全"""
    print("\n[5/5] 检查并发安全...")
    try:
        with open('src/data/collector.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'Lock()' in content and 'with self._stats_lock' in content:
            print("  ✓ collector.py使用Lock保护共享状态")
            return True
        else:
            print("  ⚠ 未检测到Lock使用")
            return False
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

def main():
    print("="*60)
    print("系统健康检查 v2.1.0-full-fix2")
    print("="*60)
    
    checks = [
        check_imports,
        check_bare_except,
        check_config_validator,
        check_execution_params,
        check_thread_safety,
    ]
    
    results = [check() for check in checks]
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ 所有检查通过 ({passed}/{total})")
        print("="*60)
        print("\n系统健康，可以使用！")
        return 0
    else:
        print(f"⚠️  部分检查失败 ({passed}/{total})")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit(main())
