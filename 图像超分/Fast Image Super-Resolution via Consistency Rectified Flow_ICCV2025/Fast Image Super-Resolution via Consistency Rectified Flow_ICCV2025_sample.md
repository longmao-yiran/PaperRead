# Fast Image Super-Resolution via Consistency Rectified Flow - 简要总结

## 核心创新点是什么

FlowSR 将真实图像超分建模为从 LR 图像直接流向 HR 图像的 Rectified Flow，并用 HR 正则化一致性学习把多步恢复能力蒸馏到一步。HR 正则化显式约束蒸馏预测回到真实 HR，修正普通一致性蒸馏可能继承教师误差的问题；快慢双时间调度则结合大跨度单步训练与细粒度流场监督。

## 主要实验结论或研究结论是什么

在 RealSR、DRealSR 和 DIV2K-Val 上，FlowSR 以一步推理取得有竞争力的感知质量和保真度。在 DRealSR 上，模型运行时间为 0.14 秒；消融表明，直接 LR→HR 流、HR 正则化一致性和快慢调度均对单步视觉质量有关键贡献。
