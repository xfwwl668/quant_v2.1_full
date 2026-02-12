#!/usr/bin/env python3
"""
测试RSRS因子NaN处理修复
验证np.nanmean是否正确处理窗口内的NaN值
"""

import numpy as np
import sys
from pathlib import Path

# 添加到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.factors.technical.rsrs import compute_rsrs_batch


def test_rsrs_nan_fix():
    """测试RSRS因子计算是否正确处理NaN"""
    print("=" * 70)
    print("RSRS因子NaN处理修复验证")
    print("=" * 70)

    # 模拟数据：20股票 × 90天
    n_stocks = 20
    n_days = 90

    print(f"\n测试数据规模: {n_stocks} 股票 × {n_days} 天 = {n_stocks * n_days} 数据点")

    # 生成模拟价格数据（包含一些NaN模拟停牌等）
    np.random.seed(42)
    base_price = 10.0 + np.cumsum(np.random.randn(n_stocks, n_days) * 0.02, axis=1)
    high = base_price * (1.0 + np.abs(np.random.randn(n_stocks, n_days)) * 0.01)
    low = base_price * (1.0 - np.abs(np.random.randn(n_stocks, n_days)) * 0.01)

    # 模拟停牌：随机设置5%的数据为NaN
    mask = np.random.random((n_stocks, n_days)) < 0.05
    high[mask] = np.nan
    low[mask] = np.nan

    # 确保C-contiguous
    high = np.ascontiguousarray(high)
    low = np.ascontiguousarray(low)

    print(f"  NaN比例: {np.sum(mask) / mask.size * 100:.1f}%")

    # 计算RSRS因子
    print("\n计算RSRS因子...")
    # ✅ FIX: 使用合理的zscore_window，确保能计算出足够的rsrs_adaptive值
    # 对于90天的数据，使用20-30的zscore_window更合适
    zscore_window = min(30, n_days)
    rsrs_beta, rsrs_r2, rsrs_std, rsrs_adaptive, rsrs_valid, rsrs_momentum = compute_rsrs_batch(
        high, low, window=18, zscore_window=zscore_window
    )

    # 统计结果
    print("\n因子计算结果:")
    print(f"  rsrs_beta:     shape={rsrs_beta.shape}, 非NaN={np.sum(~np.isnan(rsrs_beta))}/{rsrs_beta.size} ({np.sum(~np.isnan(rsrs_beta)) / rsrs_beta.size * 100:.1f}%)")
    print(f"  rsrs_r2:       shape={rsrs_r2.shape}, 非NaN={np.sum(~np.isnan(rsrs_r2))}/{rsrs_r2.size} ({np.sum(~np.isnan(rsrs_r2)) / rsrs_r2.size * 100:.1f}%)")
    print(f"  rsrs_adaptive: shape={rsrs_adaptive.shape}, 非NaN={np.sum(~np.isnan(rsrs_adaptive))}/{rsrs_adaptive.size} ({np.sum(~np.isnan(rsrs_adaptive)) / rsrs_adaptive.size * 100:.1f}%)")
    print(f"  rsrs_valid:    shape={rsrs_valid.shape}, 有效={np.sum(rsrs_valid > 0)}/{rsrs_valid.size} ({np.sum(rsrs_valid > 0) / rsrs_valid.size * 100:.1f}%)")

    # 验证通过标准
    non_nan_ratio = np.sum(~np.isnan(rsrs_adaptive)) / rsrs_adaptive.size * 100

    print("\n" + "=" * 70)
    print("验证结果:")
    print("=" * 70)

    # 检查1: 非NaN比例应该>50%
    if non_nan_ratio > 50.0:
        print(f"✅ PASS: rsrs_adaptive非NaN比例 = {non_nan_ratio:.1f}% > 50%")
        result1 = True
    else:
        print(f"❌ FAIL: rsrs_adaptive非NaN比例 = {non_nan_ratio:.1f}% <= 50%")
        result1 = False

    # 检查2: 确保没有全NaN的因子
    all_nan_beta = np.all(np.isnan(rsrs_beta))
    all_nan_r2 = np.all(np.isnan(rsrs_r2))
    all_nan_adaptive = np.all(np.isnan(rsrs_adaptive))

    if not all_nan_beta and not all_nan_r2 and not all_nan_adaptive:
        print(f"✅ PASS: 因子包含有效数据（非全NaN）")
        result2 = True
    else:
        print(f"❌ FAIL: 因子全为NaN")
        result2 = False

    # 检查3: 检查是否有合理的值范围
    if not all_nan_adaptive:
        valid_adaptive = rsrs_adaptive[~np.isnan(rsrs_adaptive)]
        print(f"✅ INFO: rsrs_adaptive有效值范围: [{np.min(valid_adaptive):.4f}, {np.max(valid_adaptive):.4f}]")
        result3 = True
    else:
        print(f"❌ FAIL: 无法统计有效值范围")
        result3 = False

    print("\n" + "=" * 70)

    if result1 and result2 and result3:
        print("✅ 所有检查通过 - RSRS因子NaN处理修复成功！")
        print("=" * 70)
        return 0
    else:
        print("❌ 部分检查失败 - RSRS因子NaN处理修复失败")
        print("=" * 70)
        return 1


def test_nanmean_vs_mean():
    """演示np.mean和np.nanmean的区别"""
    print("\n" + "=" * 70)
    print("np.mean vs np.nanmean 对比测试")
    print("=" * 70)

    # 创建包含NaN的数组
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    print(f"\n测试数组: {arr}")
    print(f"  (包含1个NaN)")

    mean_result = np.mean(arr)
    nanmean_result = np.nanmean(arr)

    print(f"\nnp.mean(arr)    = {mean_result}")
    print(f"np.nanmean(arr) = {nanmean_result}")
    print(f"\n说明:")
    print(f"  - np.mean(): 遇到NaN返回NaN")
    print(f"  - np.nanmean(): 忽略NaN计算均值")

    print("\n修复原理:")
    print("  前window-1天被设为NaN后，后续滑动窗口包含这些NaN")
    print("  np.mean() → 返回NaN → 归一化失败")
    print("  np.nanmean() → 忽略NaN → 归一化成功")

    print("=" * 70)


if __name__ == "__main__":
    # 先演示np.mean vs np.nanmean的区别
    test_nanmean_vs_mean()

    # 测试RSRS因子修复
    exit(test_rsrs_nan_fix())
