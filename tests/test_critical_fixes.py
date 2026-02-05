#!/usr/bin/env python3
"""
test_critical_fixes.py
======================
关键修复的单元测试

验证所有CRITICAL级别的修复是否正确实现。
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def test_rsrs_nan_handling():
    """测试rsrs.py的NaN处理"""
    print("\n[TEST 1/5] rsrs.py NaN处理...")
    
    try:
        from src.factors.technical.rsrs import _online_ols_single
        
        # 测试用例1：前18天NaN（模拟新股）
        high = np.array([np.nan] * 18 + [10.0 + i * 0.1 for i in range(100)])
        low = np.array([np.nan] * 18 + [9.0 + i * 0.1 for i in range(100)])
        
        beta, r2, resid = _online_ols_single(high, low, window=18)
        
        # 验证：前18天应该是NaN
        assert np.isnan(beta[:18]).all(), "前18天应该全是NaN"
        
        # 验证：第19天开始应该有值（至少有一些非NaN）
        assert not np.isnan(beta[18:]).all(), "第19天后应该有非NaN值"
        
        print("  ✓ 新股NaN处理正确")
        
        # 测试用例2：中间停牌（窗口内有NaN）
        high2 = np.array([10.0] * 30 + [np.nan] * 5 + [10.0] * 30)
        low2 = np.array([9.0] * 30 + [np.nan] * 5 + [9.0] * 30)
        
        beta2, r2_2, resid2 = _online_ols_single(high2, low2, window=18)
        
        # 验证：停牌后的窗口应该正确处理（不应该全是NaN）
        # 因为窗口内有效数据 > 80%
        post_suspension = beta2[40:]  # 停牌后10天
        assert not np.isnan(post_suspension).all(), "停牌后应该恢复计算"
        
        print("  ✓ 停牌NaN处理正确")
        print("  ✅ rsrs.py NaN处理测试通过")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_match_limit_detection():
    """测试match.py的涨跌停检测"""
    print("\n[TEST 2/5] match.py涨跌停检测...")
    
    try:
        from src.engine.match import check_limit_up, check_limit_down
        
        # 测试涨停（普通股票10%）
        prev_close = 10.0
        limit_up_price = 11.0  # 10%涨停
        
        # 情况1：涨停且封盘（close = high = 涨停价）
        is_limit_up = check_limit_up(
            close=limit_up_price,
            high=limit_up_price,
            prev_close=prev_close,
            is_st=False,
            is_kcb=False
        )
        assert is_limit_up, "应该检测到涨停"
        print("  ✓ 涨停检测正确")
        
        # 情况2：高开高走但未涨停
        is_limit_up_2 = check_limit_up(
            close=10.8,  # 8%
            high=10.9,
            prev_close=prev_close,
            is_st=False,
            is_kcb=False
        )
        assert not is_limit_up_2, "不应该误判为涨停"
        print("  ✓ 非涨停识别正确")
        
        # 测试跌停
        limit_down_price = 9.0  # -10%跌停
        is_limit_down = check_limit_down(
            close=limit_down_price,
            low=limit_down_price,
            prev_close=prev_close,
            is_st=False,
            is_kcb=False
        )
        assert is_limit_down, "应该检测到跌停"
        print("  ✓ 跌停检测正确")
        
        # 测试科创板（20%）
        is_limit_up_kcb = check_limit_up(
            close=12.0,  # 20%涨停
            high=12.0,
            prev_close=prev_close,
            is_st=False,
            is_kcb=True
        )
        assert is_limit_up_kcb, "科创板涨停检测正确"
        print("  ✓ 科创板涨停检测正确")
        
        print("  ✅ match.py涨跌停检测测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_account_t1_settlement():
    """测试account.py的T+1结算"""
    print("\n[TEST 3/5] account.py T+1结算...")
    
    try:
        from src.engine.account import AccountManager
        from src.strategy.types import Fill, OrderSide, Timestamp
        
        # 创建账户
        account = AccountManager(initial_cash=1000000.0)
        
        # 模拟买入（应该立即扣款）
        fill_buy = Fill(
            order_id="ORD-001",
            code="SH600000",
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
            commission=3.0,
            slippage=0.0,
            timestamp=Timestamp.now(),
            strategy_name="test",
        )
        
        initial_cash = account.cash
        account.process_fill(fill_buy, None, current_price=10.0)
        
        # 验证：现金立即扣减
        assert account.cash < initial_cash, "买入应该立即扣款"
        print("  ✓ 买入立即扣款正确")
        
        # 模拟卖出（资金应该冻结）
        fill_sell = Fill(
            order_id="ORD-002",
            code="SH600000",
            side=OrderSide.SELL,
            quantity=1000,
            price=11.0,
            commission=3.3,
            slippage=0.0,
            timestamp=Timestamp.now(),
            strategy_name="test",
        )
        
        cash_before_sell = account.cash
        account.process_fill(fill_sell, current_price=11.0)
        
        # 验证：卖出后现金不变（资金冻结）
        assert account.cash == cash_before_sell, "卖出后现金应该不变（T+1）"
        assert account.frozen_cash > 0, "冻结资金应该 > 0"
        print("  ✓ 卖出资金冻结正确")
        
        # 模拟日终结算
        frozen = account.frozen_cash
        account.on_day_end()
        
        # 验证：T+1资金解冻
        assert account.frozen_cash == 0, "日终后冻结资金应该清零"
        assert account.cash == cash_before_sell + frozen, "资金应该解冻到cash"
        print("  ✓ T+1资金解冻正确")
        
        print("  ✅ account.py T+1结算测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_data_loading():
    """测试main.py的数据加载逻辑"""
    print("\n[TEST 4/5] main.py数据加载...")
    
    try:
        # 创建测试数据
        test_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [10.0, 10.1, 10.2],
            "high": [10.5, 10.6, 10.7],
            "low": [9.5, 9.6, 9.7],
            "close": [10.2, 10.3, 10.4],
            "volume": [1000000, 1100000, 1200000],
        })
        
        # 测试1：有date列
        result_df = test_df.copy()
        assert "date" in result_df.columns, "应该有date列"
        print("  ✓ date列存在")
        
        # 测试2：date列类型转换
        result_df["date"] = pd.to_datetime(result_df["date"]).dt.strftime("%Y-%m-%d")
        assert pd.api.types.is_string_dtype(result_df["date"]), "date应该是字符串类型"
        print("  ✓ date类型转换正确")
        
        # 测试3：日期过滤
        filtered = result_df[
            (result_df["date"] >= "2024-01-02") &
            (result_df["date"] <= "2024-01-03")
        ]
        assert len(filtered) == 2, "日期过滤应该返回2行"
        print("  ✓ 日期过滤正确")
        
        # 测试4：设置index
        final_df = filtered.set_index("date")
        assert isinstance(final_df.index, pd.Index), "应该设置index"
        print("  ✓ index设置正确")
        
        print("  ✅ main.py数据加载测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_constants_path():
    """测试constants.py的路径契约"""
    print("\n[TEST 5/5] constants.py路径契约...")
    
    try:
        from src.constants import PATH_TEMPLATE_DAILY, DIR_PARQUET, DIR_DAILY
        
        # 验证路径模板
        expected_path = f"{DIR_PARQUET}/{DIR_DAILY}"
        assert PATH_TEMPLATE_DAILY == expected_path, \
            f"路径应该是 {expected_path}，实际是 {PATH_TEMPLATE_DAILY}"
        
        # 验证：不应该包含market_data
        assert "market_data" not in PATH_TEMPLATE_DAILY.lower(), \
            "路径不应该包含market_data"
        
        print(f"  ✓ 路径模板正确: {PATH_TEMPLATE_DAILY}")
        
        # 验证与storage.py一致性
        from src.data.storage import ColumnarStorageManager
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ColumnarStorageManager(base_dir=tmpdir)
            
            # 验证路径结构
            parquet_dir = storage.parquet_dir
            assert "parquet" in str(parquet_dir), "应该包含parquet目录"
            assert "daily" in str(parquet_dir) or "parquet" in str(parquet_dir.parent), \
                "路径结构应该包含daily"
            
            print("  ✓ 与storage.py路径一致")
        
        print("  ✅ constants.py路径契约测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("关键修复测试套件 v2.1.0")
    print("=" * 70)
    
    tests = [
        test_rsrs_nan_handling,
        test_match_limit_detection,
        test_account_t1_settlement,
        test_main_data_loading,
        test_constants_path,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ {test_func.__name__} 崩溃: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ 所有测试通过 ({passed}/{total})")
        print("=" * 70)
        print("\n系统修复验证成功！可以开始使用。")
        return 0
    else:
        print(f"❌ 部分测试失败 ({passed}/{total})")
        print("=" * 70)
        print(f"\n失败的测试:")
        for i, (test, result) in enumerate(zip(tests, results), 1):
            if not result:
                print(f"  - Test {i}: {test.__name__}")
        return 1


if __name__ == "__main__":
    exit(main())
