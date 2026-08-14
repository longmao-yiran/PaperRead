# Reverse Convolution and Its Applications to Image Restoration：简要总结

## 核心创新点是什么

论文提出 Converse2D，一种针对深度卷积的反向卷积算子。作者把恢复原特征的问题写成带二次正则的最小二乘问题，并利用 FFT 得到单步闭式解；随后将其与 LayerNorm、1×1 卷积、GELU 和残差连接组成 Converse Block，形成可用于去噪、超分辨率和去模糊的 ConverseNet 系列网络。

## 主要实验结论或研究结论是什么

在高斯去噪中，Converse-DnCNN 在 Set12/BSD68 上达到 30.70/29.36 dB，优于对应的深度卷积和深度转置卷积版本；在 ×4 超分辨率中，Converse2D 作为上采样模块的性能与 Pixelshuffle、插值等方法相当；在非盲去模糊中，核条件 Converse-USRNet 稳定优于 Conv-USRNet。结果说明 Converse2D 能作为一种具有明确逆问题解释的通用网络基础算子，但其边界假设、计算开销和复杂真实退化下的泛化能力仍需进一步验证。
