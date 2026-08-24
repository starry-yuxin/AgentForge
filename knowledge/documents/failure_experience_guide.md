# Failure Experience Guide

> 项目内置示例材料：本文档为 AgentForge 演示而编写，不代表外部行业标准。

## MissingValueError

症状：训练时报错输入包含 NaN。修复：对数值列使用 median imputation，对类别列使用 most-frequent imputation，并在训练 Pipeline 内拟合。

## CategoricalEncodingError

症状：模型无法转换字符串或遇到未知类别。修复：加入 One-Hot Encoding，并设置 `handle_unknown="ignore"`。

## LowMinorityRecall

症状：总体 accuracy 尚可，但少数类 recall 很低。修复：使用 class weight、检查分层切分，并在 validation set 上调整阈值。

## LowF1AtDefaultThreshold

症状：0.5 阈值下 F1 较低。修复：只使用 validation 标签和概率搜索 F1 最优阈值，再对 test 进行一次最终评价。

## DataLeakageRisk

症状：测试结果异常乐观或预处理使用全量数据。修复：隔离 train/validation/test，将预处理放入 Pipeline，禁止测试集参与拟合、阈值选择和候选选择。

