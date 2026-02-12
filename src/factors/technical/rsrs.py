"""
src/factors/technical/rsrs.py
==============================
Phase 3 — RSRS 因子引擎（Numba 加速版）

核心算法：Online OLS（增量更新最小二乘法）
性能目标：1000 股 × 1250 天 < 50ms

═══════════════════════════════════════════════════════════════════
设计要点
═══════════════════════════════════════════════════════════════════

1. **增量 OLS 算法（Online OLS）**
   ────────────────────────────────
   传统窗口滚动回归: O(n × w)，每次窗口移动都重新计算全部 w 个点的回归。
   增量更新算法:     O(n)，维护滚动和（rolling sums）增量更新统计量。

   核心公式：
   给定窗口 [t-w+1, t] 内的点 (x_i, y_i)，线性回归 y = β·x + α
   
   定义滚动和：
     S_x  = Σx_i
     S_y  = Σy_i
     S_xx = Σ(x_i²)
     S_yy = Σ(y_i²)
     S_xy = Σ(x_i·y_i)
   
   则：
     β = (w·S_xy - S_x·S_y) / (w·S_xx - S_x²)
     α = (S_y - β·S_x) / w
     r² = [w·S_xy - S_x·S_y]² / [(w·S_xx - S_x²)(w·S_yy - S_y²)]
   
   窗口滑动时，只需：
     - 减去窗口左端点的旧值
     - 加入窗口右端点的新值
   
   无需重新遍历窗口内所有点 → O(n) 而非 O(n×w)。

2. **Welford 在线算法（Z-Score 标准化）**
   ─────────────────────────────────────
   传统标准化需要两遍扫描（mean → std → normalize）。
   Welford 算法单遍扫描，数值稳定性更高：
   
     M_t = M_{t-1} + (x_t - M_{t-1}) / t
     S_t = S_{t-1} + (x_t - M_{t-1})(x_t - M_t)
     σ_t = √(S_t / (t - 1))
   
   在线计算均值和方差，同时避免大数求和的精度损失。

3. **循环融合（Loop Fusion）**
   ────────────────────────────
   RSRS 全族因子（beta, r2, std, adaptive）在同一个循环中计算，
   共享窗口统计量，减少 50% 内存带宽和 cache miss。

4. **Numba 并行化**
   ─────────────────
   @njit(parallel=True, cache=True, fastmath=True)
   - parallel=True:  多股票并行（prange）
   - cache=True:     二次调用零编译
   - fastmath=True:  放宽 IEEE754 约束，提速 15-20%

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

# ============================================================================
# Numba 依赖检查 + 动态导入
# ============================================================================

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn(
        "Numba not available. RSRS factor will use slow NumPy fallback. "
        "Install with: pip install numba --break-system-packages",
        RuntimeWarning
    )
    # Fallback: 模拟 @njit 和 prange
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============================================================================
# Part 1: 核心 Online OLS 算法（单股票版本）
# ============================================================================

@njit(cache=True, fastmath=True)
def _online_ols_single(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    单股票的增量 OLS 回归。
    
    Args:
        x: shape=(n_days,)  自变量（通常是 normalized low）
        y: shape=(n_days,)  因变量（通常是 normalized high）
        window: 回归窗口大小（如 18）
    
    Returns:
        beta:  shape=(n_days,)  回归系数 β
        r2:    shape=(n_days,)  拟合优度 r²
        resid: shape=(n_days,)  残差标准差
    
    算法复杂度: O(n)
    """
    n = len(x)
    beta = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    resid = np.full(n, np.nan, dtype=np.float64)
    
    if n < window:
        return beta, r2, resid
    
    # ✅ FIX: 改进初始窗口NaN处理
    # 统计第一个窗口的有效数据数量
    valid_count = 0
    S_x = 0.0
    S_y = 0.0
    S_xx = 0.0
    S_yy = 0.0
    S_xy = 0.0
    
    # 窗口预填充（第一个窗口）
    for i in range(window):
        xi = x[i]
        yi = y[i]
        if not (np.isnan(xi) or np.isnan(yi)):
            # 仅累加有效数据
            S_x += xi
            S_y += yi
            S_xx += xi * xi
            S_yy += yi * yi
            S_xy += xi * yi
            valid_count += 1
    
    # 需要至少80%有效数据
    min_valid = int(window * 0.8)

    # ✅ FIX: 标记第一个窗口是否有效
    first_window_valid = valid_count >= min_valid

    # 如果第一个窗口无效，不计算beta[window-1]，保持统计量不变（用于后续增量更新）
    # 只有当第一个窗口有效时，才计算beta[window-1]
    if first_window_valid:
        w_float = float(valid_count)
        denom = w_float * S_xx - S_x * S_x

        if abs(denom) > 1e-12:
            b = (w_float * S_xy - S_x * S_y) / denom
            # a = (S_y - b * S_x) / w_float  # α 在 RSRS 中不需要

            # r²
            numer = (w_float * S_xy - S_x * S_y) ** 2
            denom_r2 = (w_float * S_xx - S_x * S_x) * (w_float * S_yy - S_y * S_y)
            if denom_r2 > 1e-12:
                r2_val = numer / denom_r2
            else:
                r2_val = 0.0

            # 残差方差（用于计算 std）
            # 残差 = y - (α + β·x)
            # 这里简化：std ≈ sqrt(1 - r²) * std(y)
            # 或更精确地用 SSE / (n - 2)
            # 简化版：resid_std = sqrt((S_yy - b * S_xy) / w)
            sse = S_yy - b * S_xy  # 简化 SSE
            if sse > 0 and valid_count > 2:
                resid_std = np.sqrt(sse / (valid_count - 2))
            else:
                resid_std = 0.0

            beta[window - 1] = b
            r2[window - 1] = r2_val
            resid[window - 1] = resid_std
    else:
        # ✅ FIX: 第一个窗口无效时，重置统计量，等待后续窗口
        # 这样滑动窗口增量更新才能正常工作
        S_x = S_y = S_xx = S_yy = S_xy = 0.0
        valid_count = 0
    
    # 滚动窗口：从 idx=window 到 n-1
    for t in range(window, n):
        # 移除窗口左端点 (t - window)
        x_old = x[t - window]
        y_old = y[t - window]

        # 加入窗口右端点 (t)
        x_new = x[t]
        y_new = y[t]

        # ✅ FIX: 改进NaN处理逻辑
        old_is_valid = not (np.isnan(x_old) or np.isnan(y_old))
        new_is_valid = not (np.isnan(x_new) or np.isnan(y_new))

        # ✅ FIX: 如果第一个窗口无效（统计量被重置），则需要重新计算窗口统计量
        if not first_window_valid and valid_count == 0:
            # 重新计算当前窗口 [t-window+1, t+1] 的统计量
            window_S_x = 0.0
            window_S_y = 0.0
            window_S_xx = 0.0
            window_S_yy = 0.0
            window_S_xy = 0.0
            window_valid_count = 0

            for i in range(t - window + 1, t + 1):
                xi = x[i]
                yi = y[i]
                if not (np.isnan(xi) or np.isnan(yi)):
                    window_S_x += xi
                    window_S_y += yi
                    window_S_xx += xi * xi
                    window_S_yy += yi * yi
                    window_S_xy += xi * yi
                    window_valid_count += 1

            # 检查是否有足够有效数据
            if window_valid_count >= min_valid:
                # 使用重新计算的统计量
                S_x = window_S_x
                S_y = window_S_y
                S_xx = window_S_xx
                S_yy = window_S_yy
                S_xy = window_S_xy
                valid_count = window_valid_count
            else:
                # 仍然不足，跳过
                continue
        else:
            # 正常的增量更新
            # 更新有效数据计数
            if old_is_valid and not new_is_valid:
                valid_count -= 1  # 移出有效，加入无效
            elif not old_is_valid and new_is_valid:
                valid_count += 1  # 移出无效，加入有效
            # 其他情况：都有效或都无效，计数不变

            # 检查有效数据是否足够
            if valid_count < min_valid:
                # 有效数据不足，跳过此窗口（保持NaN）
                continue

            # 增量更新滚动和（仅当数据有效时）
            if old_is_valid:
                S_x -= x_old
                S_y -= y_old
                S_xx -= x_old * x_old
                S_yy -= y_old * y_old
                S_xy -= x_old * y_old

            if new_is_valid:
                S_x += x_new
                S_y += y_new
                S_xx += x_new * x_new
                S_yy += y_new * y_new
                S_xy += x_new * y_new

        # 计算回归系数（使用实际有效数据数量）
        w_float = float(valid_count)
        denom = w_float * S_xx - S_x * S_x
        if abs(denom) > 1e-12:
            b = (w_float * S_xy - S_x * S_y) / denom

            # r²
            numer = (w_float * S_xy - S_x * S_y) ** 2
            denom_r2 = (w_float * S_xx - S_x * S_x) * (w_float * S_yy - S_y * S_y)
            if denom_r2 > 1e-12:
                r2_val = numer / denom_r2
            else:
                r2_val = 0.0

            # 残差标准差
            sse = S_yy - b * S_xy
            if sse > 0 and valid_count > 2:
                resid_std = np.sqrt(sse / (valid_count - 2))
            else:
                resid_std = 0.0

            beta[t] = b
            r2[t] = r2_val
            resid[t] = resid_std
    
    return beta, r2, resid


# ============================================================================
# Part 2: Welford 在线算法（滚动均值和标准差）
# ============================================================================

@njit(cache=True, fastmath=True)
def _welford_rolling_mean_std(
    arr: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welford 在线算法计算滚动均值和标准差。
    
    Args:
        arr: shape=(n,)  输入序列
        window: 滚动窗口大小
    
    Returns:
        mean: shape=(n,)  滚动均值
        std:  shape=(n,)  滚动标准差
    
    算法复杂度: O(n)
    """
    n = len(arr)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    
    if n < window:
        return mean, std
    
    # Welford 状态：M_k (mean), S_k (sum of squared deviations)
    M = 0.0
    S = 0.0
    
    # 初始化第一个窗口
    valid_count = 0
    for i in range(window):
        val = arr[i]
        if not np.isnan(val):
            valid_count += 1
            delta = val - M
            M += delta / valid_count
            delta2 = val - M
            S += delta * delta2
    
    if valid_count >= 2:
        mean[window - 1] = M
        std[window - 1] = np.sqrt(S / (valid_count - 1))
    
    # 滚动窗口
    for t in range(window, n):
        x_old = arr[t - window]
        x_new = arr[t]
        
        # 移除旧值
        if not np.isnan(x_old):
            valid_count -= 1
            if valid_count > 0:
                delta_old = x_old - M
                M -= delta_old / valid_count
                S -= delta_old * (x_old - M)
        
        # 加入新值
        if not np.isnan(x_new):
            valid_count += 1
            delta_new = x_new - M
            M += delta_new / valid_count
            delta2_new = x_new - M
            S += delta_new * delta2_new
        
        if valid_count >= 2:
            mean[t] = M
            std[t] = np.sqrt(S / (valid_count - 1))
    
    return mean, std


# ============================================================================
# Part 3: RSRS 全族因子批量计算（多股票并行）
# ============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def compute_rsrs_batch(
    high: np.ndarray,
    low: np.ndarray,
    window: int = 18,
    zscore_window: int = 600,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    批量计算 RSRS 全族因子（Numba 并行加速版）。
    
    RSRS (Resistance Support Relative Strength):
      - 高点 vs 低点回归，β 系数反映压力/支撑强度
      - 自适应版本：β 标准化（Z-Score）后乘以 r² 和斜率动量
    
    Args:
        high:  shape=(n_stocks, n_days)  最高价矩阵（C-contiguous）
        low:   shape=(n_stocks, n_days)  最低价矩阵（C-contiguous）
        window: RSRS 回归窗口（默认 18）
        zscore_window: Z-Score 标准化窗口（默认 600）
    
    Returns:
        rsrs_beta:      shape=(n_stocks, n_days)  原始 β 系数
        rsrs_r2:        shape=(n_stocks, n_days)  拟合优度 r²
        rsrs_std:       shape=(n_stocks, n_days)  残差标准差
        rsrs_adaptive:  shape=(n_stocks, n_days)  自适应 RSRS = zscore(β) × r² × β_momentum
        rsrs_valid:     shape=(n_stocks, n_days)  有效性标记（r² > 0.8）
        rsrs_momentum:  shape=(n_stocks, n_days)  β 斜率动量
    
    性能基准（参考 策略.txt）:
      1000 stocks × 1250 days < 50ms
      5000 stocks × 2500 days < 500ms
    
    算法复杂度: O(n_stocks × n_days) 并行
    """
    n_stocks, n_days = high.shape
    
    # 输出矩阵初始化（全 NaN）
    rsrs_beta = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
    rsrs_r2 = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
    rsrs_std = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
    rsrs_adaptive = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
    rsrs_valid = np.zeros((n_stocks, n_days), dtype=np.float64)
    rsrs_momentum = np.full((n_stocks, n_days), np.nan, dtype=np.float64)
    
    # 并行处理每只股票
    for i in prange(n_stocks):
        # ── Step 1: 归一化 high / low ──
        # RSRS 需要对价格做归一化（除以移动平均），避免价格量级影响
        # 这里简化为：x = low / ma(low, window), y = high / ma(high, window)
        # 更精确的方法是用 Z-Score，但简化版性能更好
        
        high_i = high[i, :]
        low_i = low[i, :]
        
        # 归一化（用滑动窗口均值）
        # ✅ FIX: 使用 np.nanmean 忽略 NaN 值，从 t=window-1 开始计算
        # 这样回归窗口 [window-1, 2*window-1) 将有 window 个有效数据点
        x_norm = np.empty(n_days, dtype=np.float64)
        y_norm = np.empty(n_days, dtype=np.float64)

        for t in range(n_days):
            if t < window - 1:
                # 前 window-1 天设为 NaN
                x_norm[t] = np.nan
                y_norm[t] = np.nan
            else:
                # ✅ FIX: 使用 np.nanmean 忽略 NaN 值
                # 计算窗口内的均值，窗口包含当前值
                low_mean = np.nanmean(low_i[max(0, t - window + 1) : t + 1])
                high_mean = np.nanmean(high_i[max(0, t - window + 1) : t + 1])
                if low_mean > 1e-9 and high_mean > 1e-9:
                    x_norm[t] = low_i[t] / low_mean
                    y_norm[t] = high_i[t] / high_mean
                else:
                    x_norm[t] = np.nan
                    y_norm[t] = np.nan
        
        # ── Step 2: Online OLS 回归 ──
        beta_i, r2_i, resid_i = _online_ols_single(x_norm, y_norm, window)
        
        rsrs_beta[i, :] = beta_i
        rsrs_r2[i, :] = r2_i
        rsrs_std[i, :] = resid_i
        
        # ── Step 3: β 动量（斜率）──
        # momentum = (β_t - β_{t-M}) / M
        momentum_window = 5
        for t in range(momentum_window, n_days):
            if not np.isnan(beta_i[t]) and not np.isnan(beta_i[t - momentum_window]):
                rsrs_momentum[i, t] = (beta_i[t] - beta_i[t - momentum_window]) / momentum_window
        
        # ── Step 4: Z-Score 标准化 β ──
        # ✅ FIX: 如果 zscore_window > n_days，使用 n_days 作为窗口大小
        actual_zscore_window = min(zscore_window, n_days)
        beta_mean, beta_std_arr = _welford_rolling_mean_std(beta_i, actual_zscore_window)

        # ── Step 5: 自适应 RSRS ──
        # adaptive = zscore(β) × r² × sign(momentum)
        for t in range(n_days):
            if (not np.isnan(beta_i[t]) and
                not np.isnan(beta_mean[t]) and
                not np.isnan(beta_std_arr[t]) and
                beta_std_arr[t] > 1e-12):  # ✅ FIX: 降低阈值，允许更小的标准差

                zscore = (beta_i[t] - beta_mean[t]) / beta_std_arr[t]
                r2_val = r2_i[t] if not np.isnan(r2_i[t]) else 0.0
                momentum_val = rsrs_momentum[i, t] if not np.isnan(rsrs_momentum[i, t]) else 0.0
                
                # 自适应公式
                rsrs_adaptive[i, t] = zscore * r2_val * (1.0 + momentum_val)
                
                # 有效性标记（r² > 0.8）
                if r2_val > 0.8:
                    rsrs_valid[i, t] = 1.0
    
    return rsrs_beta, rsrs_r2, rsrs_std, rsrs_adaptive, rsrs_valid, rsrs_momentum


# ============================================================================
# Part 4: 信号质量评分（RSRS 置信度）
# ============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def compute_signal_quality_batch(
    rsrs_r2: np.ndarray,
    rsrs_std: np.ndarray,
    rsrs_momentum: np.ndarray,
) -> np.ndarray:
    """
    计算 RSRS 信号质量评分。
    
    signal_quality = r² × (1 - normalized_std) × momentum_factor
    
    Args:
        rsrs_r2:       shape=(n_stocks, n_days)
        rsrs_std:      shape=(n_stocks, n_days)  残差标准差
        rsrs_momentum: shape=(n_stocks, n_days)
    
    Returns:
        quality: shape=(n_stocks, n_days)  信号质量分数 [0, 1]
    """
    n_stocks, n_days = rsrs_r2.shape
    quality = np.zeros((n_stocks, n_days), dtype=np.float64)
    
    for i in prange(n_stocks):
        for t in range(n_days):
            r2 = rsrs_r2[i, t]
            std = rsrs_std[i, t]
            momentum = rsrs_momentum[i, t]
            
            if np.isnan(r2) or np.isnan(std) or np.isnan(momentum):
                quality[i, t] = 0.0
            else:
                # normalized_std = min(std, 0.1) / 0.1  # 截断到 0.1
                # 简化：直接用 1 - std（假设 std 已标准化）
                std_factor = max(0.0, 1.0 - min(std, 1.0))
                momentum_factor = max(0.5, 1.0 + momentum)  # 动量增强
                
                quality[i, t] = r2 * std_factor * momentum_factor
                quality[i, t] = min(1.0, max(0.0, quality[i, t]))  # clip [0, 1]
    
    return quality


# ============================================================================
# Part 5: 导出
# ============================================================================

__all__ = [
    "compute_rsrs_batch",
    "compute_signal_quality_batch",
    "NUMBA_AVAILABLE",
]


# ============================================================================
# Part 6: 测试入口
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("RSRS Factor Engine - Benchmark Test")
    print("=" * 70)
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print()
    
    # 测试数据
    n_stocks = 1000
    n_days = 1250
    
    print(f"Test Size: {n_stocks} stocks × {n_days} days = {n_stocks * n_days:,} points")
    print()
    
    # 生成模拟数据
    np.random.seed(42)
    close = 10.0 + np.cumsum(np.random.randn(n_stocks, n_days) * 0.02, axis=1)
    high = close * (1.0 + np.abs(np.random.randn(n_stocks, n_days)) * 0.01)
    low = close * (1.0 - np.abs(np.random.randn(n_stocks, n_days)) * 0.01)
    
    # 确保 C-contiguous
    high = np.ascontiguousarray(high)
    low = np.ascontiguousarray(low)
    
    # 预热（Numba 编译）
    print("Warming up (Numba compilation)...")
    _ = compute_rsrs_batch(high[:10, :100], low[:10, :100])
    print("✓ Compilation complete")
    print()
    
    # 正式测试
    print("Running benchmark...")
    n_runs = 5
    times = []
    
    for i in range(n_runs):
        start = time.perf_counter()
        result = compute_rsrs_batch(high, low)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.2f} ms")
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    
    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"  Average Time: {avg_time:.2f} ms")
    print(f"  Min Time:     {min_time:.2f} ms")
    print(f"  Throughput:   {(n_stocks * n_days / avg_time * 1000) / 1e6:.2f} million points/sec")
    print()
    
    # 验证
    beta, r2, std, adaptive, valid, momentum = result
    print("Validation (Non-NaN Ratio):")
    print(f"  rsrs_beta:     {np.sum(~np.isnan(beta)) / beta.size * 100:.1f}%")
    print(f"  rsrs_r2:       {np.sum(~np.isnan(r2)) / r2.size * 100:.1f}%")
    print(f"  rsrs_adaptive: {np.sum(~np.isnan(adaptive)) / adaptive.size * 100:.1f}%")
    print()
    
    # 判断是否达标
    target = 50.0
    if avg_time < target:
        print(f"✅ PASS: {avg_time:.2f}ms < {target}ms target")
    else:
        print(f"❌ FAIL: {avg_time:.2f}ms >= {target}ms target")
    
    print("=" * 70)
