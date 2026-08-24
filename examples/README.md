# 可复制示例

以下示例由 `python scripts/export_demo_artifacts.py` 从当前确定性工作流导出，不包含密钥、用户目录绝对路径、模型二进制、完整日志或付费 API 调用。

```text
examples/
├── requests/churn_request.txt
├── generated/logistic_regression.py
├── generated/random_forest.py
├── reports/workflow_report.md
├── reports/workflow_report.json
└── repair_case/
    ├── README.md
    └── repair.diff
```

## 默认中文需求

```text
请构建客户流失预测模型，比较 Logistic Regression 和 Random Forest，以 F1 作为主要指标，最低要求为 0.60，数据包含缺失值、类别特征和类别不平衡。
```

直接运行内置等价请求：

```bash
python -m agentforge.cli demo
```

## 英文自定义请求

```bash
python -m agentforge.cli run \
  --request "Build a customer churn binary classifier using Logistic Regression and Random Forest; primary metric F1, minimum 0.60." \
  --dataset data/churn_sample.csv \
  --metric f1 \
  --minimum-score 0.60 \
  --no-persist
```

## 确定性修复闭环

```bash
python -m agentforge.cli demo --inject-failure missing_imputer
```

该命令故意让 Logistic Regression 的 attempt-0 缺少插补步骤，用于展示真实失败、分类、知识经验检索、差异文件和 attempt-1 修复。它不会调用 LLM。
