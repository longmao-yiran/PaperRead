# AP-BSN 论文阅读示例

## 基本信息

- 标题：AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network
- 会议：CVPR 2022
- 类型：自监督图像去噪

## 核心结论

论文使用训练阶段的 PD5 与推理阶段的 PD2，分别兼顾真实噪声去相关与图像细节保留；随后以随机替换细化（R3）降低残余噪声相关性。该方法只需要真实噪声 sRGB 图像，不需要干净参考图或噪声先验。

完整结构化阅读报告见：[AP-BSN- Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network_CVPR2022.md](AP-BSN-%20Self-Supervised%20Denoising%20for%20Real-World%20Images%20via%20Asymmetric%20PD%20and%20Blind-Spot%20Network_CVPR2022.md)
