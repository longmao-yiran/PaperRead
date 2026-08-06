# Back to Basics: Let Denoising Generative Models Denoise - 简要总结

## 核心创新点是什么

论文区分了扩散模型的“损失空间”和“网络直接预测空间”，并提出在高维像素 token、有限网络容量下，应让网络直接预测处于低维流形的干净图像，而不是直接预测近似全维的噪声或速度。基于该思想，作者提出 JiT：无 tokenizer、无预训练、直接以大像素 patch 为输入的普通 Transformer 扩散模型。

## 主要实验结论或研究结论是什么

ImageNet 实验表明，大 patch 条件下图像直接预测显著优于直接噪声或速度预测；后两者会发生严重 FID 退化。即使 patch 维度大于 Transformer hidden size，JiT 仍可稳定生成，高效 bottleneck 还可能提升质量。JiT-G 在 ImageNet 256 和 512 上分别达到 FID 1.82 与 1.78。
