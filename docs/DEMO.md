# AgentForge 3–5 分钟演示脚本

所有现场命令默认使用 deterministic 模式，不读取 `.env`、不调用外部 LLM。

## 0:00–0:30 项目介绍

AgentForge 是一个多 Agent 算法能力工厂原型：它把自然语言需求变成结构化计划，结合知识图谱生成两个算法实现，通过安全与接口检查后独立执行，以 validation 指标选优、test 指标做最终报告，并把成功验证摘要回写图谱。

## 0:30–1:00 架构

打开 README 的两张 Mermaid 图，依次说明 Requirement、Knowledge、Planner、Code、Validation、Repair、Report、Persistence 八个 Agent。强调 LLM 是可选增强，默认 Demo 不产生 API 费用。

## 1:00–2:00 运行完整 Demo

```bash
python -m agentforge.cli demo
```

重点展示：

1. `mode: deterministic` 与 `llm_calls: 0`；
2. Logistic Regression 和 Random Forest 两个计划；
3. 每个候选的 AST、接口和 subprocess 状态；
4. validation F1 用于选择，test F1/ROC-AUC 只做最终报告；
5. `outputs/runs/<run-id>/generated`、`models`、`results`、`reports`。

## 2:00–2:45 展示代码与报告

打开任一 `candidate.py`，展示统一的 `train`、`predict`、`evaluate` 接口。随后打开 `workflow_report.md`，展示结构化需求、知识来源、候选指标、执行事件和安全限制。

## 2:45–3:30 展示自动修复

```bash
python -m agentforge.cli demo --inject-failure missing_imputer
```

说明 attempt-0 真实触发缺失值错误，ErrorClassifier 识别 `MissingValueError`，RepairAgent 检索 FailureExperience，attempt-1 恢复插补并重新通过验证。静态脱敏 diff 位于 `examples/repair_case/repair.diff`。

## 3:30–4:00 展示知识图谱

```bash
python scripts/query_knowledge.py \
  --task binary_classification \
  --metric f1 \
  --characteristic missing_values
python scripts/visualize_knowledge_graph.py
```

展示 `docs/assets/knowledge-graph.svg`。这是从真实 JSON 图谱生成的代表性子图。正式运行若开启持久化，会新增 ValidationRun 并通过 `USED_ALGORITHM`、`PERFORMED_ON`、`ACHIEVED_METRIC` 等关系回写。

## 4:00–4:30 可选 LLM 与安全边界

项目支持 OpenAI Responses 与 OpenAI-compatible Chat Completions。DeepSeek 的 HTTP/SDK 连接和 `system` 角色兼容已真实验证；空值、展示名称和新增任务别名通过离线回归测试。不要声称最后一次真实结构化解析已完整成功。

AST 与 subprocess 只能降低意外风险，不是容器或生产级沙箱，不应执行任意不可信代码。

## 常见问题

1. **为什么不用 test 选模型？** 防止对最终评价集过拟合；候选和阈值只看 validation。
2. **代码真的是生成的吗？** 默认是根据计划实例化的确定性可执行模板；LLM 自由生成默认关闭。
3. **为什么选择 NetworkX？** 离线、零服务依赖，适合小规模笔试原型与 GraphML/JSON 交付。
4. **能否执行任意 LLM 代码？** 不建议；现有 AST 与 subprocess 不是生产沙箱。
5. **修复是否无限循环？** 否，最多两轮，且仅处理白名单失败类型。
6. **DeepSeek 是否完全验证？** 连接和消息角色已真实验证；最新别名补充只有离线回归验证。
7. **没有 Key 能否演示？** 可以，裸 demo 是完整 deterministic 工作流。
8. **为什么有两个报告格式？** JSON 供机器审计，Markdown 供人阅读。
9. **指标能代表业务效果吗？** 不能，当前使用固定种子的合成数据，只证明流程正确和可复现。
10. **下一步怎么扩展？** 可增加受控算法插件、其他任务类型、服务接口和更强资源隔离，但不属于当前交付。
