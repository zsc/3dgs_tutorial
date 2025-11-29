# 第 8 章 · 生成式与扩散 3DGS：DreamGaussian, DiffusionGS 与 DiffGS

## 1. 开篇：从“重建现实”到“想象现实”

在前面的章节中，我们的核心逻辑一直是 **逆向渲染（Inverse Rendering）**：给定一组拍摄好的真实照片，通过优化高斯参数，试图“还原”出场景的 3D 结构。这被称为“重建（Reconstruction）”。

然而，AI 领域的圣杯在于 **生成（Generation）**。能否仅凭一句文本描述（"一只穿着宇航服的柯基犬"）或一张单视图图片，就从零创造出一个完整的、可 360 度观看的 3D 高斯场景？

在 2023 年之前，DreamFusion (NeRF + Diffusion) 虽然证明了这条路可行，但生成一个物体往往需要数小时。NeRF 昂贵的体积渲染成为了巨大的拦路虎。**3D Gaussian Splatting 的出现引发了生成式 3D 的速度革命**。其毫秒级的光栅化能力，使得梯度回传的效率提升了几个数量级，将 Text-to-3D 的时间从“小时级”压缩到了“分钟级”甚至“秒级”。

本章将带你进入这一激动人心的领域。我们将探讨 3DGS 如何作为 3D 表征与 Stable Diffusion 等 2D 扩散模型结合，从基于优化的 **DreamGaussian** 到基于前馈大模型的 **LGM/DiffusionGS**，并最终解决如何将那一团“高斯云”转化为工业界可用的 Mesh 网格。

**本章学习目标：**
1.  **深入理解 SDS 与 VSD**：掌握利用 2D 扩散模型“蒸馏”出 3D 几何的核心数学原理。
2.  **掌握 DreamGaussian 管线**：学习“生成—精修”两阶段策略，理解如何平衡生成速度与网格质量。
3.  **区分优化法与前馈法**：理解 Per-scene Optimization（优）与 Feed-forward Models（推理）的本质区别及适用场景。
4.  **解决几何与纹理问题**：学习如何处理 Janus（多头）问题、烘焙光照问题以及网格提取技术。

---

## 2. 核心原理：跨维度的“蒸馏”

生成式 3D 的本质是：**我们没有 3D 真值（Ground Truth），只有 2D 的先验知识。** 我们需要一个机制，让 2D 扩散模型充当“老师”，指导 3DGS 这个“学生”不断变形，直到它在任何角度看起来都符合描述。

### 2.1 灵魂公式：SDS (Score Distillation Sampling)

SDS 是 DreamFusion 提出的核心贡献，也是目前 90% 生成式 3DGS 的基石。

**直观理解**：
想象你捏了一个泥人（随机初始化的 3DGS），你从某个角度拍了一张照给老师（Stable Diffusion）看。老师说：“这看着不像柯基，这里的像素噪点应该往‘毛发’的方向调整。”老师给出的不是具体的修改指令，而是对图像噪声的预测（Score）。我们将这个“修改方向”转化为梯度，反向传播给 3D 高斯球的位置、颜色和不透明度。

**数学表达**：
SDS 损失函数的梯度计算如下：

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} \triangleq \mathbb{E}_{t, \epsilon} \left[ w(t) (\hat{\epsilon}_\phi(\mathbf{x}_t; y, t) - \epsilon) \frac{\partial \mathbf{x}}{\partial \theta} \right]
$$

其中各项含义至关重要：
*   $\theta$：3D 高斯的参数（均值 $\mu$, 协方差 $\Sigma$, 颜色 $c$, 不透明度 $\alpha$）。
*   $\mathbf{x}$：3DGS 在当前视角渲染出的图像。
*   $\mathbf{x}_t$：向图像 $\mathbf{x}$ 添加了强度为 $t$ 的噪声后的图像。
*   $\hat{\epsilon}_\phi(\dots)$：预训练的 2D 扩散模型（如 SD）预测出的噪声。
*   $\epsilon$：实际添加的随机噪声。
*   $y$：文本描述（Prompt）的嵌入向量。
*   $\frac{\partial \mathbf{x}}{\partial \theta}$：**雅可比矩阵**，即渲染图像对高斯参数的导数。这正是 3DGS 的强项——相比 NeRF这个导数的计算极快且显存占用低。

> **Rule of Thumb (SDS 的特性)**：SDS 倾向于产生“过饱和”和“平滑”的结果。这是因为 SDS 实际上是在寻找概率密度的高地（Mode Seeking），常常忽略高频纹理细节。在 3DGS 中，这可能导致生成的物体看起来像塑料玩具。

### 2.2 进阶：VSD (Variational Score Distillation)

为了解决 SDS 的过饱和与细节丢失问题，后续工作（如 ProlificDreamer）提出了 VSD。虽然计算开销更大，但值得了解。

VSD 将 3D 参数也视为一个分布，并同时训练一个“粒子 LoRA”来模拟当前的 3D 分布。简单来说，SDS 是 3D 试图模仿 2D 模型的通用先验，而 VSD 是 3D 试图模仿一个“在这个特定物体上微调过的”2D 分布。对于追求极致质量的 3DGS 生成，VSD 及其变体是 2024-2025 年的研究热点。

---

## 3. 优化流派：DreamGaussian 与两阶段法

**DreamGaussian** 是将 3DGS 用于生成任务的代表作它之所以能在 2024 年引起轰动，是因为它将 Image-to-3D 的时间从 1 小时缩短到了 2 分钟。

### 3.1 阶段一：生成粗糙高斯 (Generative Coarse Stage)

在这个阶段，目标是快速建立几何轮廓。

1.  **初始化**：
    *   **Text-to-3D**：在一个球体内随机采样点作为初始高斯中心。
    *   **Image-to-3D**：利用现成的单目深度估计（如 Depth Anything）或多视角生成模型（如 Zero-1-to-3），反投影得到初始点云。**这一点至关重要**，良好的初始化能避免 90% 的形状崩坏。
2.  **训练循环**：
    *   随机采样相机视角（方位角、仰角）。
    *   渲染图像 + 混合背景（通常在黑、白之间随机切换，防止高斯学习背景色）。
    *   计算 SDS Loss + 透明度正则项（Opacity Regularization）。
    *   **致密化与剪枝**：DreamGaussian 采用了比原始 3DGS 更激进的致密化策略。因为是从零生长，需要高斯球快速填满物体内。频率通常是每 100 步就进行一次 split/clone。

### 3.2 阶段二：网格精修 (Mesh Refinement Stage)

直接生成的 3DGS 往往像一团模糊的云，无法直接用于游戏引擎。
DreamGaussian 引入了巧妙的转换管线：

1.  **高斯转网格**：
    *   从 3DGS 中提取等值面（Iso-surface）。由于高斯是离散的，这通常通过查询体素网格的密度来实现。
    *   使用 Marching Cubes 算法提取 Mesh。
    *   **去噪与平滑**：提取出的 Mesh 表面通常极其粗糙（布满小颗粒），需要进行拉普拉斯平滑（Laplacian Smoothing）或重网格化（Remeshing）。
2.  **UV 展开与纹理烘焙**：
    *   对 Mesh 进行自动 UV 展开（如 xatlas）。
    *   将 3DGS 渲染的图像逆向投影回 UV 贴图，作为初始纹理。
3.  **纹理微调**：
    *   固定 Mesh 几何，只优化纹理贴图。
    *   再次使用 SDS Loss 或 MSE Loss（如果有参考图），但在 **UV 空间** 进行超分辨率处。这一步可以把模糊的高斯颜色变成高清的 2K 贴图。

---

## 4. 前馈与大模型流派：DiffusionGS, LGM 与 AnySplat

优化方法（Optimization-based）虽然好，但每个物体都要重新训练几分钟，无法做到“即时生成”。
2024-2025 年的趋势是 **前馈模型（Feed-forward Models）**：输入一张图，直接由神经网络输出所有高斯参数。

### 4.1 3DGS 的表示难题

将 3DGS 放入神经网络（如 U-Net 或 Transformer）极其困难，因为：
1.  **无序性**：高斯是一个集合（Set），没有固定的顺序。
2.  **数量可变**：一个复杂的树可能需要 10 万个高斯，一个简单的盒子只需要 100 个。

### 4.2 解决方案：结构化 3DGS

为了让神经网络能“吐出”高斯，主流方法有以下几种：

1.  **Pixel-aligned Gaussians (像素对齐高斯)**：
    *   代表作：**LGM (Large Gaussian Model)**, **AnySplat**。
    *   **原理**：将 Image-to-3D 视为 Image-to-Image 任务。络输出的是一张多通道图像，每个像素对应一个高斯的属性（xyz 偏移, opacity, color, scale, quaternion）。
    *   输入图片分辨率为 $H \times W$，则直接生成 $H \times W$ 个高斯。
    *   利用 Plucker Ray embedding 来处理多视角几何。
    *   **优势**：速度极快（< 0.1秒），完全兼容现有的 CNN/Transformer 架构。

2.  **Voxel / Tri-plane Diffusion**：
    *   代表作：**DiffusionGS** (部分变体), **Rodin** (NeRF变体)。
    *   **原理**：先训练一个 Autoencoder，将无序的高斯压缩到规则的 Tri-plane（三平面特征）或 Latent Voxel 中。
    *   在 Latent Space 上训练 Diffusion Model。
    *   **优势**：可以生成从未见过的拓扑结构，不仅限于像素对应的表面。

### 4.3 生成管线的融合 (Hybrid Pipeline)

最先进的流程（截至 2025）通常是混合的：
1.  使用 **LGM** 等前馈模型在 1 秒内生成一个高质量的初始 3DGS。
2.  以此为初始化，进行短时间的 **SDS/VSD 微调**，补充被前馈模型平滑掉的高频细节。

---

## 5. 关键技术深潜：从云雾到实体

生成式 3DGS 最常见的失败是生成了“全息影像”——看着很美，侧面一看全是半透明薄片。如何保证生成的是“实体”？

### 5.1 熵正则化 (Entropy Regularization)

为了迫使不透明度 $\alpha$ 二值化（要么是 0，要么是 1），我们需要在 Loss 中加入熵约束：

$$
\mathcal{L}_{\text{opacity}} = \lambda \sum_{i} -\left[ \alpha_i \log \alpha_i + (1-\alpha_i) \log (1-\alpha_i) \right]
$$

或者更简单的：$\mathcal{L} = \sum (\alpha_i)^2 + (1-\alpha_i)^2$ 的变体。
这一步对于后续提取 Mesh 至关重要。

### 5.2 尺度正则化 (Scale Regularization)

高斯倾向于变大变扁来“作弊”覆盖更大的屏幕区域。必须惩罚高斯的 Scaling 乘积或最大边长：

$$
\mathcal{L}_{\text{scale}} = \sum_{i} \prod_{j=1}^3 s_{i,j} \quad \text{(惩罚体积)}
$$

或者限制长轴与短轴的比例，防止生成过扁的“面片”。

### 5.3 SuGaR 与表面对齐

**SuGaR (Surface Gaussian Reconstruction)** 提出了一种通过正则项强制高斯分布在物体表面的方法。它引入了一个正则项，使得高斯的中心尽可能贴近深度图推导出的表面，并且高斯的扁平方向与表面切平面一致。
在生成任务中，常将 SuGaR 的正则项作为 Loss 的一部分，以便直接生成可提取 Mesh 的高斯场。

---

## 6. 本章小结

*   **速度即正义**：3DGS 极大地加速了 SDS 优化过程，使得 Text-to-3D 变得即时可用。
*   **两条路线并进**：
    *   **优化法 (DreamGaussian)**：灵活、质量上限高，适合精细创作。
    *   **前馈法 (LGM/AnySplat)**：毫秒级生成，适合大规模资产库构建。
*   **几何质量是核心**：生成好看的图片容易，生成几何正确的 Mesh 很难。必须依赖 Opacity/Scale 正则化和深度图约束。
*   **从 3DGS 到 Mesh**：网格提取 + UV 纹理烘焙是 3DGS 融入传统图形学工作流（Unity/Unreal）的必经之路。

---

## 7. 常见陷阱与错误 (Gotchas)

### 1. Janus Problem（双面怪/多头问题）
**现象**：生成的角色后脑勺上还有一张脸，或者马有 8 条腿。
**原因**：2D 扩散模型在训练时见过很多“正面”，它认为“背面”看起来也应该像“正面”。它缺乏 3D 全局一致性。
**调试技巧**：
*   **Prompt Engineering**：渲染背面视角时，强制修改 prompt（如 append `", back view"`），渲染侧面时 append `", side view"`。
*   **负向 Prompt**：加入 `multi-head, extra limbs, bad anatomy`。
*   **缩小方位角范围**：如果在做 Image-to-3D，限制相机的采样范围，不要让相机转到对于单图来说“不可见”的极端角度太久，除非有先验填补。

### 2. Floaters（漂浮物/伪影）
**现象**：物体周围漂浮着许多细小的彩色高斯点，像灰尘一样。
**原因**：SDS Loss 在空旷区域梯度噪声导致的误判。
**调试技巧**：
*   **激进的剪枝**：在训练后期，提高 opacity threshold，直接杀掉所有 $\alpha < 0.1$ 的高斯。
*   **基于连通域的后处理**：训练结束后，计算高斯的 K-近邻，删除那些“孤立”的高斯团簇。

### 3. 只有颜色没有几何（The Billboard Effect）
**现象**：物体看起来是 3D 的，但转动视角发现它其实是几层画了画的纸片拼起来的。
**原因**：深度正则化不足，或者初始化不好。
**调试技巧**：
*   **使用 Depth Loss**：如果输入是图片，务必使用 Depth Anything V2 预测深度图，并计算 $\mathcal{L}_{\text{depth}}$。这是强制几何正确的唯一强约束。
*   **随机背景**：不要只在黑色背景下训练，随机切换红/蓝/白/格纹背景，迫使高斯学习正确的轮廓而不是把背景画在物体边缘。

### 4. 光照烘焙（Baked Lighting）问题
**现象**：生成的 Mesh 自带阴影和高光（例如鼻子下有黑影），放入游戏引擎打光后效果非常怪异。
**原因**：SD 生成的图是带光影的，3DGS 忠实地把光影当成了颜色（Albedo）学了进去。
**调试技巧**：
*   **Relighting 模型**：使用专门微调过的去光照 SD 模型。
*   **基于着色的渲染 (Shading-based Rendering)**：在训练后期，引入简单的 Lambertian 光照模型，强制网络预测 Albedo 和 Normal，而不是直接预测最终颜色。 $\text{Color} = \text{Albedo} \times (\text{Normal} \cdot \text{Light})$。

---

**下一章预告**：
解决了“生成”问题后，我们如何让这些 3DGS 理解它们“是什么”？下一章我们将探讨 **第 9 章 · 语言与语义 3DGS**，学习如何将 CLIP 和 SAM 特征嵌入高斯场，实现“在 3D 空间中用自然语言搜索物体”。
