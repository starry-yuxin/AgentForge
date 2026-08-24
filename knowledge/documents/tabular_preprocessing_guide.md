# Tabular Preprocessing Guide

> 项目内置示例材料：本文档为 AgentForge 演示而编写，不代表外部行业标准。

## Feature type identification

表格二分类首先区分数值特征与类别特征。标识符和目标列不得作为训练特征。

## Missing values

数值缺失值可使用训练集统计得到的中位数填补；类别缺失值可使用训练集众数填补。预处理器必须只在训练集拟合。

## One-Hot Encoding

无序类别特征适合 One-Hot Encoding。对未知类别应采用忽略策略，避免验证或线上数据导致运行失败。

## Feature scaling

Logistic Regression 等线性模型通常受益于 Standard Scaling。树模型通常不要求缩放，但可共享一致的预处理接口。

## Avoiding data leakage

任何插补、编码、缩放和阈值选择都不能使用测试集标签或全量数据统计量。应通过 Pipeline 将预处理拟合限定在训练集。

## Train, validation, and test responsibilities

Train 用于拟合模型和预处理器；validation 用于模型选择与分类阈值选择；test 只用于方案固定后的最终一次评价。

