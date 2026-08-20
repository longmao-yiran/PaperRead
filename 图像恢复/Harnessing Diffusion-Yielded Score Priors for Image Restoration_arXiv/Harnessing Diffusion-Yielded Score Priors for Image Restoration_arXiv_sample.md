# Harnessing Diffusion-Yielded Score Priors for Image Restoration：简要总结

## 核心创新点是什么

HYPIR 将预训练扩散模型的 score prior 作为恢复 GAN 的初始化，然后仅对中间 U-Net 的 LoRA 参数进行 GAN、LPIPS 和 MSE 联合微调。它不使用扩散损失、教师蒸馏或多步采样，因此以一次前向推理实现生成式图像恢复；同时可继承文本提示、纹理强度和随机采样等控制能力。

## 主要实验结论或研究结论是什么

作者报告扩散初始化使对抗训练的初始梯度更小、模式坍塌更弱、约 10k 步即可收敛，并在主观评价及多项感知质量指标上取得强结果。Flux 初始化版本在 DIV2K 上 LPIPS 最优，但 PSNR/SSIM 未必最优，说明该方法优先改善视觉真实感而非严格像素保真；文本与随机控制也会带来生成式幻觉风险。
