# 量化回测系统 v2.1.0-full-fix2

**企业级量化回测框架 - 紧急修复版**

[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-2.1.0--full--fix2-blue.svg)]()
[![Strategies](https://img.shields.io/badge/strategies-6-orange.svg)]()

---

## 🚀 快速开始

```bash
# 1. 解压
tar -xzf quant_v2.1_full_fix2.tar.gz
cd quant_v2.1_full

# 2. 安装
pip install -r requirements.txt

# 3. 健康检查
python health_check.py

# 4. 运行
python main.py
```

---

## 🔧 本次修复 (fix2)

### 紧急修复
1. **✅ HybridExecutionEngine参数错误**
   - 移除不支持的`slippage_base`参数
   - 修复TypeError

2. **✅ 裸except修复（6处）**
   - collector.py: 5处
   - short_term.py: 1处
   - 改为`except Exception as e:`

3. **✅ 配置验证集成**
   - 新增ConfigValidator
   - 启动时自动验证
   - 防止无效配置

4. **✅ 并发安全审计**
   - 确认Lock使用正确
   - 共享状态受保护

---

## 📊 6个策略

| 策略 | 类型 | 风险 | 适用 |
|-----|------|------|------|
| RSRSMomentumStrategy | 中长线 | 低 | 牛市 |
| RSRSAdvancedStrategy | 中长线 | 中 | 震荡 |
| AlphaHunterStrategy | 高频 | 中 | 牛市 |
| ShortTermStrategy | 短线 | 低 | 所有 |
| MomentumReversalStrategy | 组合 | 中 | 震荡 |
| SentimentReversalStrategy | 反转 | 高 | 熊市 |

---

## 💻 使用示例

### 单策略
```python
from src.strategy.strategies import AlphaHunterStrategy
from src.engine.execution import HybridExecutionEngine

strategy = AlphaHunterStrategy(top_n=15)
engine = HybridExecutionEngine(
    strategy=strategy,
    initial_cash=1000000.0,
    start_date="2024-01-01",
    end_date="2024-06-30",
)
result = engine.run_backtest(history)
```

### 多策略对比
```bash
python compare_strategies.py
```

---

## 🔍 健康检查

```bash
python health_check.py
```

**检查项**:
- ✅ 策略导入
- ✅ 裸except
- ✅ 配置验证
- ✅ 引擎参数
- ✅ 并发安全

---

## 📁 核心文件

```
quant_v2.1_full/
├── health_check.py           # 健康检查 ⭐
├── compare_strategies.py     # 策略对比
├── main.py                    # 主入口 (已修复)
├── CHANGELOG.md               # 更新日志 ⭐
├── src/
│   ├── config_validator.py   # 配置验证 ⭐
│   ├── engine/
│   │   └── execution.py      # (参数已修复)
│   └── strategy/strategies/
│       ├── rsrs_strategy.py
│       ├── rsrs_advanced.py
│       ├── alpha_hunter.py
│       ├── short_term.py     # (裸except已修复)
│       ├── momentum_reversal.py
│       └── sentiment_reversal.py
└── data/                      # 数据目录
```

---

## ⚠️ 风险声明

### 高风险场景（禁止）
- ❌ 真实资金实盘
- ❌ 高频交易（ms级）
- ❌ 多租户SaaS

### 适用场景
- ✅ 回测研究
- ✅ 策略开发
- ✅ 因子挖掘
- ✅ 学习教学

---

## 📝 已知限制

1. 涨跌停检测使用open价估算prev_close（误差2-3%）
2. 未实现完整T+1持仓冻结
3. 时区为naive datetime
4. 无实盘接口

---

## 📚 文档

- **CHANGELOG.md** - 详细更新日志
- **QUICK_START.md** - 快速上手
- **FIXES_v2.1.0.md** - 原始修复清单

---

## 🎯 优先级行动

### Week 1 (已完成) ✅
- [x] 并发安全
- [x] 裸except
- [x] 配置验证
- [x] 参数修复

### Week 2-3 (进行中)
- [ ] 单元测试 (60%覆盖)
- [ ] Prometheus监控
- [ ] 文档完善

### Month 2 (规划)
- [ ] CI/CD
- [ ] 批量撮合优化
- [ ] 日志升级

---

**版本**: v2.1.0-full-fix2  
**状态**: 生产就绪 ✅  
**健康检查**: `python health_check.py`  

立即开始: `python main.py`
