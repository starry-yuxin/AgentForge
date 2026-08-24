# Classification Model Guide

> 项目内置示例材料：本文档为 AgentForge 演示而编写，不代表外部行业标准。

## Logistic Regression

Logistic Regression 适合需要概率输出、快速训练和较强可解释性的表格二分类任务。数值特征应适当缩放，类别特征应编码。

## Random Forest

Random Forest 适合非线性关系和特征交互明显的表格任务，对缩放不敏感，并能处理较复杂的决策边界。

## Class imbalance and class weight

类别不平衡时，默认决策规则可能造成少数类召回率过低。Logistic Regression 可使用 `class_weight="balanced"`；Random Forest 可使用 `class_weight="balanced_subsample"`。

## F1 and ROC-AUC

F1 综合 precision 与 recall，适合关注少数类识别质量的场景。ROC-AUC 必须由连续预测概率计算，不能由二值预测计算。

## Validation-based threshold selection

分类阈值应在 validation set 上按目标指标搜索。test set 不得参与阈值或候选算法选择。

