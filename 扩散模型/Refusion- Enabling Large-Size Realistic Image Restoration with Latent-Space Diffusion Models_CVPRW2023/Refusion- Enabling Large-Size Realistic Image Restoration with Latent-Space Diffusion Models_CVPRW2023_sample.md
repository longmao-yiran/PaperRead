# Refusion 简要总结

## 核心创新点是什么？

Refusion 在 IR-SDE 均值回归扩散框架上，以时间条件化 NAFNet 改进噪声预测，并提出带 U-Net 跳连的潜空间扩散：只在低分辨率 latent 中执行多步去噪，同时将原低质量图像的多尺度 hidden feature 交给解码器。其 Maximum-Likelihood 目标监督随机时间步的最优反向落点，而非直接比较最终图像或只拟合瞬时噪声；latent-replacing 预训练则不依赖 VAE-GAN 或对抗损失。

## 主要实验结论或研究结论是什么？

Refusion 在立体超分和去阴影中显著改善 LPIPS/FID 等感知指标；去阴影挑战总榜第 2 且 LPIPS、MOS 最佳。Latent Refusion 将扩散计算由 63.4G FLOPs 降至 4.0G，并能处理完整 6000×4000 非均匀去雾图像。与 FLUX.2 的速度型 Flow Matching 相比，IR-SDE 的落点监督更贴合成对保真复原，但更依赖 LQ/HQ 配对、均值回归假设和离散噪声日程。
