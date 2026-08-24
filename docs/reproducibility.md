# AgentForge 全量复现指南

本指南只使用确定性模式，不读取 `.env`、不调用外部 LLM，也不会启用 LLM 代码生成或修复。

## 1. 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python --version
```

项目声明支持 Python `>=3.11,<3.15`。所有依赖范围同时维护在 `requirements.txt` 和 `pyproject.toml`。

## 2. 数据与基础 Benchmark

仓库已经包含固定种子生成的 `data/churn_sample.csv`。如需从源码重建：

```bash
python scripts/generate_sample_data.py
python scripts/run_demo.py
```

`scripts/run_demo.py` 是基础模型 Benchmark；完整多 Agent 流程应使用 CLI demo。

> `python scripts/build_knowledge_graph.py` 用于从基础能力数据重建基准图谱，并会写入正式 GraphML/JSON。正式工作流产生的 ValidationRun 属于运行历史；重建前应备份图谱，或在隔离副本中验证。

## 3. 完整确定性工作流

```bash
python -m agentforge.cli demo
```

验收要点：

- `mode: deterministic`
- `llm_calls: 0`
- 两个候选均经过安全、接口和 subprocess 检查
- `best_candidate` 由 validation F1 选择
- JSON 与 Markdown 报告存在
- demo 默认 `persist=False`，不会污染正式图谱

## 4. 故障与修复演示

```bash
python -m agentforge.cli demo --inject-failure missing_imputer
```

预期 Logistic Regression 的首次执行真实失败并分类为 `MissingValueError`，随后通过知识图谱中的 `missing_value_error` 经验生成 attempt-1。Random Forest 独立继续执行。修复次数上限为两轮。

## 5. 知识检索与可视化

```bash
python scripts/query_knowledge.py \
  --task binary_classification \
  --metric f1 \
  --characteristic missing_values

python scripts/visualize_knowledge_graph.py
python scripts/export_demo_artifacts.py
```

可视化脚本固定布局种子并写入 `docs/assets/knowledge-graph.svg`。它只读取提交的 JSON 图谱，不访问网络。

## 6. 自动测试

```bash
python -m pytest -q
```

测试全部使用临时目录和 fake/mock LLM 客户端，覆盖：

- 固定种子数据生成与 train/validation/test 隔离；
- 能力 Schema、图谱往返、检索和 ValidationRun 回写；
- Agent 输入输出与完整工作流；
- AST 安全、统一接口、subprocess 超时与日志；
- 失败分类、最多两轮修复和故障候选隔离；
- LLM 配置、严格 JSON、别名规范化、provider 适配和受控 fallback。

## 7. 验收矩阵

| 检查 | 命令 | 通过标准 |
|---|---|---|
| 依赖一致性 | `python -m pip check` | `No broken requirements found` |
| 全量测试 | `python -m pytest -q` | 全部通过且无网络请求 |
| 默认安全性 | `python -m agentforge.cli demo` | `llm_calls: 0` |
| 修复闭环 | `... --inject-failure missing_imputer` | 首次失败、attempt-1 成功 |
| 图谱可视化 | `python scripts/visualize_knowledge_graph.py` | SVG 成功生成 |
| 补丁格式 | `git diff --check` | 无输出 |
| 密钥边界 | `git check-ignore .env` | 输出 `.env` |

## 8. 可复现性说明

模型与数据拆分使用固定随机种子。由于运行时间、底层 BLAS 和依赖补丁版本可能不同，耗时可能变化；在声明的依赖范围与同一数据下，选择逻辑和主要指标应保持稳定。任何实际 LLM 模式都不属于严格确定性的复现基线。

仓库没有额外增加 Makefile：现有 CLI 和五个单用途脚本已经覆盖 demo、test、graph、query 与 examples，直接命令在 macOS/Linux 上更透明，也避免引入第二套命令维护层。
