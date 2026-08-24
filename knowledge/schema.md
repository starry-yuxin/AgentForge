# AgentForge Knowledge Graph Schema

## Purpose

该图谱表达客户流失表格二分类场景中的算法、预处理、指标、约束、依赖、失败经验和验证历史，为后续 Agent 提供可解释检索结果。

## Node types

- `Task`：机器学习任务，如二分类。
- `Algorithm`：可执行或可生成的算法能力。
- `Preprocessor`：插补、编码、缩放、类别平衡和阈值优化能力。
- `Metric`：模型评价指标及其计算约束。
- `Constraint`：安全、数据隔离或运行约束。
- `Dataset`：验证使用的数据集及数据属性。
- `ValidationRun`：一次可追溯验证运行及其说明。
- `FailureExperience`：失败现象、原因和推荐修复经验。
- `Dependency`：算法或能力依赖的软件包。
- `SourceDocument`：项目内置示例知识来源，用于追溯。

## Relationship types

- `SUITABLE_FOR`：算法适用于任务。
- `REQUIRES`：算法或任务需要预处理能力。
- `EVALUATED_BY`：任务或算法使用某指标评价。
- `HANDLES`：能力或经验处理某种数据特征/问题。
- `SATISFIES`：能力满足某项约束。
- `USED_ALGORITHM`：验证运行使用了算法。
- `PERFORMED_ON`：验证运行在某数据集上执行。
- `FAILED_BECAUSE`：失败验证由某失败经验解释。
- `IMPROVED_BY`：失败经验可由某能力改善。
- `ACHIEVED_METRIC`：验证运行取得某指标值，边属性保存算法和值。
- `DEPENDS_ON`：能力依赖某软件包。
- `DERIVED_FROM`：知识节点来源于项目内置材料。

## Properties

公共节点属性包括 `node_type`、`name`、`description`、`version`、`source_document` 和 `source_section`。能力节点还可包含 `inputs`、`outputs`、`applicable_tasks`、`applicable_conditions`、`constraints`、`dependencies`、`metrics`。ValidationRun 保存时间、随机种子、最佳算法和免责声明；指标值保存在 `ACHIEVED_METRIC` 边上。

GraphML 只可靠支持标量属性，因此 list、dict 和 null 在导出前编码为带前缀的 JSON 字符串，加载时恢复原类型。可读 JSON 图谱使用 node-link 结构保存复杂属性。

## Example triples

- `LogisticRegression -[SUITABLE_FOR]-> BinaryClassification`
- `LogisticRegression -[REQUIRES]-> StandardScaling`
- `BinaryClassification -[EVALUATED_BY]-> F1`
- `LowMinorityRecall -[IMPROVED_BY]-> ClassWeightBalancing`
- `stage1_validation -[PERFORMED_ON]-> customer_churn_sample`
- `stage1_validation -[ACHIEVED_METRIC {algorithm: LogisticRegression, value: 0.6739}]-> F1`

## Why NetworkX instead of Neo4j

NetworkX 无需外部服务，适合两天原型、离线评分环境和小规模知识图谱，JSON/GraphML 文件可以随仓库直接复现。Neo4j 更适合多人协作、大规模图查询和长期在线服务，但会增加安装、认证和部署成本。

## Future responsibility split

知识图谱负责语义关系、可解释检索、能力与失败经验关联；SQLite 未来负责结构化运行事实、完整日志、代码版本和事务性历史。SQLite 是运行记录事实源，图谱保存适合检索的摘要与关系，两者通过稳定 ID 关联。

