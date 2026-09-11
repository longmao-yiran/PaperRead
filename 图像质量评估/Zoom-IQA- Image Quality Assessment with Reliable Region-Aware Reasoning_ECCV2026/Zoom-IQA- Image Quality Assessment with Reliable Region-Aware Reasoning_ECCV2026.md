# Zoom-IQA: Image Quality Assessment with Reliable Region-Aware Reasoning

## 一、论文基本信息

- **论文类型**：图像质量评估（无参考 IQA）、视觉语言模型、强化学习。
- **会议**：ECCV 2026；arXiv:2601.02918v3。
- **作者**：Guoqiang Liang、Jianyi Wang、Zhonghua Wu、Shangchen Zhou、Chen Change Loy。
- **单位**：S-Lab, Nanyang Technological University；SenseTime Research。
- **论文链接**：<https://arxiv.org/abs/2601.02918>。
- **项目主页**：<https://ethanliang99.github.io/ZOOMIQA-Projectpage/>。

## 二、摘要总结

传统无参考图像质量评估通常只预测分数，缺乏可解释性；近期 VLM 方法虽可同时生成理由和评分，但多为一次性、纯文本推理，不能在发现可疑区域时回到图像获取细节证据。Zoom-IQA 将人类“先整体判断、发现疑点、放大局部、重新确认”的过程建模为交互式视觉推理。它首先以约 7,000 条 GR-IQA 轨迹做监督微调，学习结构化质量理由、边界框和裁剪动作；再用 GRPO 强化学习学习何时裁剪和裁剪何处。为避免强化学习令评分与理由趋同，作者对高风险数值 token 施加选择性的 KL-Coverage 正则，并以渐进重采样缓解长尾评分分布。实验显示，该方法在多种真实、合成及 AI 生成图像基准上获得稳定的评分性能，且质量描述优于对比推理模型；生成的区域化理由还能改善文本引导图像恢复的感知质量。

## 三、研究背景

![图 1：现有 IQA 与 Zoom-IQA 的对比](images/fig_01_overview.png)

图 1 对比了单次、非交互式 IQA 与 Zoom-IQA。前者可能只发现局部问题，或给出与图像不符的描述；后者先提出缺陷假设、对关键区域裁剪，再以局部视觉证据确认结论。图的下半部分还展示，该理由可转化为恢复模型的文本指导。

### 3.1 已有研究进展

无参考 IQA 经历了手工特征、端到端回归网络和 VLM 三类路线。前两类方法通常仅输出分数，跨域泛化和可解释性有限。VLM-IQA 又分为重视评分的方法与重视描述的方法；Q-Insight、VisualQuality-R1 已用 RL 同时生成理由和分数，但仍只在文本空间内单次推理，缺少对局部视觉证据的主动核验。

### 3.2 具体科学问题

图像质量缺陷往往细小、局部且不在中心；但 IQA 数据集一般只有全局 MOS，缺少“人类查看了哪里、为何裁剪、局部证据怎样改变评分”的轨迹监督。论文要解决的就是：如何让 VLM 在无区域人工标注的情形下，学会依据自身不确定性选择区域、放大细节并给出可验证的质量理由。

## 四、研究方法

![图 2：Zoom-IQA 两阶段框架](images/fig_02_framework.png)

图 2 展示全流程：第一阶段通过 SFT 学会正确调用裁剪动作；第二阶段通过 RL 学会为解决质量不确定性而选择哪些区域。推理时，模型先看原图；输出 `crop` 后系统根据边界框生成局部图；第二轮再输入原图、局部图与首轮推理，输出最终评分。

### 4.1 数据来源和范围

GR-IQA 由 KonIQ 图像构建，约含 7,000 条质量推理轨迹。每条轨迹具有两部分：`<think>` 是四段理由，依次为整体质量概览、改进方向、应避免问题、决策与理由；`<answer>` 是包含边界框、评分、工具动作的 JSON。工具为 `crop` 时表示应放大局部，工具为 `final` 时直接结束。论文未说明 GR-IQA 的完整训练/验证划分、每样本允许的最大裁剪次数与训练步数。

评分回归覆盖 KonIQ、SPAQ、LIVE-Wild 三个真实拍摄数据集，KADID、PIPAL、CSIQ 三个合成失真数据集，以及 AGIQA 的 AI 生成图像。采用 PLCC 和 SRCC 衡量评分相关性。

### 4.2 阶段一：Grounded Quality Rationale Learning

![图 3：GR-IQA 数据构造](images/fig_03_dataset.png)

图 3 的数据管线包含生成、视觉依赖过滤和文本一致性过滤三步。Gemini-2.5-Pro 按结构化提示生成初始“理由—动作—评分”轨迹。若模型认为整图不足以确定质量，它必须在理由中指出疑点，并在答案中给出待裁剪边界框；裁剪后需说明局部证据使初始评分上调、下调还是维持。

#### 4.2.1 Visual Reliance Filtering（VRF）

VRF 用于筛除看似合理、实则不依赖图像的轨迹。作者比较两种条件的输出：一种输入原图和部分理由，另一种只输入相同的部分理由。若评分差异、边界框差异和输出不确定性变化都过小，说明语言先验已经足以复现答案，样本被丢弃。评分差、边界框 IoU、熵差阈值分别为 0.05、0.5、0.01。

![图 6：VRF 被过滤样本示例](images/fig_06_vrf.png)

图 6 表明，若有无图像时模型仍生成近似理由、评分和裁剪区域，便不能把该轨迹视为可靠视觉 grounding。该策略不对图像注入扰动，避免改变 IQA 原本要评价的噪声、模糊或压缩失真。

#### 4.2.2 Hint-Augmented Consistency Filtering（HACF）

VRF 主要验证最终动作是否依赖视觉，而中间理由仍可能虚构缺陷。HACF 以 Qwen-2.5-32B 充当评审器，同时输入图像、生成理由和低层质量提示，例如亮度、锐度、颜色统计。评审器输出通过或拒绝；只有通过者保留。它验证的不是“模型有没有看图”，而是“理由是否与图像事实和质量提示一致”。

![表 11：HACF 丢弃样本](images/table_11_hacf.png)

表 11 给出被 HACF 剔除的例子，说明生成理由即使语言流畅，只要与图像质量线索或标签提示冲突，也不被用于 SFT。

#### 4.2.3 Grounded Rationale Fine-Tuning

训练输入是图像和 CoT 提示，目标是完整的结构化序列，包括四段文字理由和 JSON 动作。其损失为：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)=-\sum_{t=1}^{|S|}\log p_{\theta}(S_t\mid S_{<t},I,T_{\mathrm{CoT}})
$$

该交叉熵使参数为 theta 的 VLM 在图像与提示条件下逐 token 重建目标轨迹。目标序列中的动作与边界框使模型学习“怎样”以可执行格式定位、裁剪并解释质量问题；但监督模仿本身不保证模型在新图上知道“何时”裁剪最有收益。

### 4.3 阶段二：Self-Guided Exploration

第二阶段以 GRPO 优化动态策略。每条候选轨迹既包含自然语言推理，也包含评分和裁剪决定。作者生成 8 条响应后，用格式、绝对分数与排序一致性共同评价，令高奖励轨迹更可能被采样。

#### 4.3.1 KL-Coverage Regularizer

RL 初期策略熵会下降，IQA 中表现为评分数值和理由路径塌缩。作者不对全部 token 一视同仁地施加约束，而是先计算 token 的对数概率与优势值的中心化关联：

$$
\mathrm{Cov}(y_i)=\left(\log\pi_{\theta}(y_i\mid y_{<i})-\overline{\log\pi}\right)\left(A(y_i)-\overline{A}\right)
$$

该量衡量某 token 的概率变化是否与轨迹优势强耦合。作者只考虑 `<answer>` 中代表最终评分的数值 token，将协方差最高的前 2% 标为高风险 token，因为它们最可能被奖励快速推向少数固定数字。

$$
\mathcal{L}_{\mathrm{KLC}}=\frac{\sum_{y_i\in\mathcal{N}_{ans}}M_iD_{\mathrm{KL}}\left(\pi_{\theta_{\mathrm{old}}}(y_i\mid y_{<i})\Vert\pi_{\theta}(y_i\mid y_{<i})\right)}{\sum_{y_i\in\mathcal{N}_{ans}}M_i}
$$

该式只在掩码选中的数值 token 上约束旧策略和当前策略的 KL 距离。这样能防止评分分布塌缩，又不强力限制文字理由和区域选择的探索；这正是它相对全序列 KL 的设计取舍。

#### 4.3.2 Progressive Re-sampling

质量分数呈长尾分布，中间质量图像多、极高或极低质量图像少。作者先按原分布训练以学习总体质量概念，随后逐阶段提高稀缺区间的采样频率。该策略将极端质量样本的学习放到已有整体能力之后，降低了直接过采样带来的分布失真，并提升全分数范围的鲁棒性。

#### 4.3.3 格式、分数与排序奖励

格式奖励要求四段 `<think>`、合法 `<answer>`、边界框、评分和工具字段都正确；任一格式条件不满足即为零。分数奖励鼓励预测评分靠近 MOS：

$$
R_{\mathrm{score}}=\exp\left(-\frac{(r_{\mathrm{pred}}-r_{\mathrm{gt}})^2}{2\sigma^2}\right)
$$

预测与标注分数越接近，该高斯奖励越接近 1；参数 sigma 控制误差的惩罚范围。它主要约束绝对评分精度，在真实世界数据集上尤其有帮助。

排序奖励在 batch 内比较图像对，要求模型的相对质量偏好与由 MOS 得到的真实偏好一致：

$$
R_{\mathrm{rank}}(x_i)=\frac{1}{B-1}\sum_{j\ne i}\left(\sqrt{\hat{p}_{ij}p^*_{ij}}+\sqrt{(1-\hat{p}_{ij})(1-p^*_{ij})}\right)
$$

该式比较预测偏好概率与真实偏好概率；预测偏好由 Thurstone 模型按评分分布的均值和方差计算。它确保模型不仅拟合单图的绝对值，也维持质量排序；消融表明，这一项对合成失真数据更有益。

$$
R_{\mathrm{total}}=R_{\mathrm{format}}+\alpha_{\mathrm{score}}R_{\mathrm{score}}+\alpha_{\mathrm{rank}}R_{\mathrm{rank}}
$$

总奖励将可执行格式、绝对分数和相对排序统一到 GRPO 学习信号中。实验设置中，分数项权重为 1，排序项权重为 2，体现作者对质量相对顺序的重视。

### 4.4 实现与推理行为

Backbone 是 Qwen2.5-VL-7B。SFT 使用 batch size 2、梯度累积 4、学习率 2.5 乘以 10 的负 6 次方、warm-up ratio 0.3；GRPO 使用 batch size 1、梯度累积 2、学习率 1 乘以 10 的负 6 次方、KL 系数 0.04。第一轮允许模型选择 `crop` 或 `final`；一旦有裁剪，第二轮融合原图、局部图和首轮理由，并固定输出最终动作。

![表 9：初始质量评估提示](images/table_09_prompt_initial.png)

![表 10：裁剪后的最终评估提示](images/table_10_prompt_final.png)

表 9 规定初始阶段应提出待检查的不确定区域；表 10 要求裁剪后明确引用新增证据，并说明初始评分为何上调、下调或确认不变。这使推理过程兼具自然语言可读性和系统可执行性。

## 五、图表分析

### 5.1 主文定量结果

![表 1：评分回归结果](images/table_01_regression.png)

表 1 表明，Zoom-IQA 在七个基准上具有稳定竞争力。其 PLCC 在 PIPAL、LIVE-Wild、CSIQ 分别为 0.468、0.887、0.797；SRCC 在 CSIQ 为 0.754。它并非每一列都超过专用回归器，但相对具推理能力的 Q-Insight 和 VisualQuality-R1，在多数基准取得更稳健的总体表现。

![表 2：质量描述评估](images/table_02_reasoning.png)

表 2 用 Gemini-2.5-Flash 与 GPT-5-mini 从准确性、合理性、完整性、置信度评价理由。KonIQ 上，Zoom-IQA 的 Gemini 评审四项为 8.72、8.80、8.30、8.60；SPAQ 上为 8.63、8.69、8.47、8.63，说明区域化复核尤其改善理由的可靠性。

![图 4：与推理式 IQA 的定性对比](images/fig_04_qualitative.png)

图 4 显示，竞争方法可能给出错误的全局描述；Zoom-IQA 会先标出可疑缺陷，再裁剪确认局部证据，因而更接近真实评分排名。

![表 3：推理引导图像恢复](images/table_03_restoration.png)

![图 5：恢复定性结果](images/fig_05_restoration.png)

表 3 与图 5 说明，Zoom-IQA 的文字理由可替代 SUPIR/DreamClear 的默认提示生成器。它在 SUPIR 上获得最高 MANIQA 0.5455 和 CLIPIQA 0.6773；在 DreamClear 上获得最低 NIQE 3.914、最高 MUSIQ 66.97 与 CLIPIQA 0.6946。PSNR 不最高，意味着收益主要在感知质量与细节恢复，而非逐像素保真。

### 5.2 消融与策略验证

![表 4：组件消融](images/table_04_ablation.png)

表 4 显示，加入 KL-Coverage 后，KonIQ PLCC 从 0.908 提升至 0.932，CSIQ 从 0.759 提升至 0.791；渐进训练使完整模型进一步达到 0.938。分数奖励更有利于 KonIQ、SPAQ 等真实图像，排序奖励更有利于 KADID、CSIQ 等合成失真。

![表 5：中心裁剪和单阶段 SFT 对比](images/table_05_policy.png)

![表 6：显著性 grounding](images/table_06_saliency.png)

表 5 证明主动策略优于固定中心裁剪，KonIQ PLCC 提升 0.040；也证明仅靠单阶段 SFT 不能学到有效区域选择。表 6 中，Zoom-IQA 的裁剪面积约 36% 至 37%，而基础 VLM 约 65% 至 66%，同时 SPAQ 的紧致覆盖指标从 0.06 增至 0.78，说明它不是盲目放大大区域。

### 5.3 补充实验与全部补充图表

![表 7：LLM 与人类评估](images/table_07_llm_human.png)

![表 8：推理延迟](images/table_08_runtime.png)

表 7 以额外 LLM 与人类评价交叉验证理由质量；表 8 报告多步 zoom 的代价约为 0.061 至 0.068 秒每图，高于非交互基线但仍处于较低延迟范围。

![表 12：理由评估提示](images/table_12_reasoning_eval.png)

表 12 给出 VLM-as-a-judge 的评价提示，表明评价器同时参考图像、生成理由与低层质量指标；这提高可比性，但仍存在评审模型偏好的局限。

![图 7：补充定性比较一](images/fig_07_extra_qualitative.png)

![图 8：补充定性比较二](images/fig_08_extra_qualitative.png)

![图 9：补充定性比较三](images/fig_09_extra_qualitative.png)

三幅图都强调：在复杂内容中，模型先形成缺陷假设，再用区域裁剪验证纹理、物体边缘或局部失真，避免将无关背景当成质量问题。

![图 10：复杂场景的交互推理一](images/fig_10_qualitative.png)

![图 11：复杂场景的交互推理二](images/fig_11_qualitative.png)

![图 12：简单场景的直接判断一](images/fig_12_qualitative.png)

![图 13：简单场景的直接判断二](images/fig_13_qualitative.png)

图 10、11 是复杂场景，使用“假设—裁剪—确认”闭环；图 12、13 是全局失真较明显的简单场景，模型可不裁剪而直接完成判断，说明裁剪由不确定性而非固定规则触发。

![图 14：背景散景鲁棒性](images/fig_14_bokeh.png)

![图 15：夜景背景噪声鲁棒性](images/fig_15_night_noise.png)

图 14、15 检验背景干扰。面对艺术性散景，模型聚焦人脸等主体而不把背景模糊错判成技术缺陷；面对夜景暗天空，模型也优先检查被照亮主体，避免无信息的暗背景主导评分。

## 六、主要发现

1. SFT 能教会模型结构化 grounding 和裁剪格式，但 RL 才能学出有效的区域选择时机。
2. KL-Coverage 可抑制最终评分数值塌缩；渐进重采样提升极端质量区间的可靠性。
3. 动态裁剪比中心裁剪更准确，且产生更紧致、显著性更高的区域。
4. 高质量、区域化的 IQA 理由可迁移为图像恢复模型的文本条件。

## 七、核心贡献

- 构建了经过 VRF 和 HACF 筛选的 GR-IQA 轨迹数据集。
- 将“整体判断—区域查看—最终确认”引入 VLM-IQA，形成可执行的交互式区域推理。
- 提出仅约束评分数字 token 的 KL-Coverage，针对性缓解 RL 中的评分塌缩。
- 证明 IQA 推理除评估外还能为生成式恢复提供有效感知指导。

## 八、研究局限

- 数据生成和过滤依赖 Gemini 与 Qwen 等外部模型，数据质量仍受教师模型、提示和评审偏好影响。
- 论文未完整披露 GR-IQA 的样本细节、完整训练步数与最大裁剪轮数，复现仍有门槛。
- 多步推理增加时延，批量大规模筛图时需权衡解释性与吞吐。
- 理由质量含 VLM-as-a-judge 评价，不能完全排除裁判模型偏好。
- 下游恢复的 PSNR 并非总是最好，说明该方法更应定位为感知质量与可解释性工具，而非通用像素级优化目标。

## 九、论文总结

Zoom-IQA 的价值在于把 IQA 从“一次性分数或文字输出”改造成可复查的视觉交互过程。它以高质量轨迹让模型学会如何定位和解释，再以 RL 学会何时主动获取局部证据，并用专门面向评分 token 的正则保持输出多样性。对于可解释视觉 agent、质量评估和图像恢复系统，论文的重要启发是：模型的理由应当对应可定位、可复核的视觉证据，而不是只生成看似合理的文本。
