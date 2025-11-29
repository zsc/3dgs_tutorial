# 第 10 章 · 通用/前馈 3DGS 与非配准数据：AnySplat 与场景泛化

**文件**：`chapter10.md`

## 1. 开篇：从“过拟合”到“推理”的范式跃迁

在 3DGS 出现的第一年，绝大多数工作都遵循着“**过拟合（Overfitting）**”的范式：对于每一个新场景，我们都需要从零开始初始化高斯，并运行数千次迭代来“记住”这个场景。这带来了两个无法忽视的工程痛点：
1.  **时间成本**：即便 3DGS 训练很快，用户上传视频后仍需等待数分钟。
2.  **前置依赖**：严重依赖 COLMAP 等传统 SFM（Structure-from-Motion）管线来获取相机位姿（Camera Pose）和稀疏点云。一旦面对弱纹理白墙）、大动态（人群）、或低视差（旋转拍摄）场景，COLMAP 往往会失效，导致 3DGS 无法启动。

**通用/前馈 3DGS（Generalizable / Feed-forward 3DGS）** 的出现打破了这一僵局。这类方法的核心愿景是：**训练一个通用的神经网络，使其看过成千上万个 3D 场景后，学会了“如何重建 3D”。** 当面对从未见过的图像时，它不再需要优化，而是直接通过**一次前向推理（Single Forward Pass）**，在毫秒级时间内回归出 3D 高斯参数。

与此同时，**AnySplat** 等工作（2024-2025）进一步结合了 **DUSt3R** 等几何基础模型，试图彻底移除对 COLMAP 的依赖，实现从“一堆乱序照片”到“高质量 3D 场景”的端到端生成。

**本章学习目标**：
- 深入理解 **pixelSplat**、**MVSplat** 等前馈网络的核心架构（Cost Volume vs. Epipolar Attention）。
- 掌握从 2D 特征图回归 3D 高斯属性（位置、旋转、缩放、球谐）的数映射机制。
- 剖析 **AnySplat** 如何利用几何大模型解决非配准（Unposed）数据的重建。
- 理解联合优化（Joint Optimization）在无 Pose 场景下的关键作用。

---

## 2. 核心原理：前馈网络的解剖学

在前馈范式中，我们的目标是学习一个映射函数 $\mathcal{F}$，输入 $N$ 张源图像 $\{I_k\}_{k=1}^N$ 及其相对位姿（在通用方法中通常假设已知，在 Unposed 方法中需同时预测），直接输出全场景的高斯集合 $\mathcal{G}$。

$$
\mathcal{G} = \{ (\mu_i, \Sigma_i, \alpha_i, c_i) \} = \mathcal{F}_\theta(I_1, \dots, I_N, P_1, \dots, P_N)
$$

这个函数 $\mathcal{F}_\theta$ 通常由深度神经网络实现，其典型架构包含三个阶段：**特征提取**、**多视图融合**、**高斯参数解码**。

### 2.1 像素对齐高斯（Pixel-aligned Gaussian）假设
为了将 2D CNN/Transformer 与 3D 空间联系起来，绝大多数前馈方法采用了“像素对齐”策略：
> **假设**：输入图像特征图（Feature Map）上的每一个“像素”或“网格点”，都对应 3D 空间中的一条射线，并在特定的深度 $d$ 处生成一个或多个 3D 高斯。

这意味着网络的首要任务不是直接预测 $(x,y,z)$，而是**预测深度 $d$**。一旦有了深度，结合相机内参 $K$ 和外参 $R, t$，高斯中心 $\mu$ 就被确定了：
$$
\mu_{uv} = R^T ( d_{uv} \cdot K^{-1} [u, v, 1]^T - t )
$$
这种设计将复杂的 3D 定位问题降维成了经典的**深度估计（Depth Estimation）**问题。

### 2.2 架构之争：极线注意力 vs 代价体 (Cost Volume)

为了准确估计深度，网络必须理解多视图几何。目前主要有两派技术路线：

#### A. 极线注意力机制 (Epipolar Transformer)
代表作：**pixelSplat** (CVPR 2024)。
- **原理**：对于参考图中的像素 $u$，网络不仅查看自身的上下文，还利用 Cross-Attention 去查询其他视角图像的特征。
- **几何约束**：为了减少计算量，Attention 被限制在**极线（Epipolar Line）**附近。即像素 $u$ 只会去“关注”与其几何上可能对应的区域。
- **优点**：能够处理宽基线（Wide-baseline）的大视角变化。
- **缺点**：Transformer 的计算复杂度较高，显存占用大，推理速度相对较慢（~1fps 级别）。

#### B. 平面扫描代价体 (Plane-Sweeping Cost Volume)
代表作：**MVSplat** (ECCV 2024), **GPS-Gaussian**。
- **原理**：借用经典 MVS（Multi-View Stereo）的思想。
    1.  在 3D 空间假设 $D$ 个深度平面。
    2.  将源视图的特征图投影（Homography Warping）到这些平面上，构建一个 3D 代价体（Cost Volume）。
    3.  使用 3D CNN 对代价体进行正则化，通过 Softmax 得到深度概率分布。
- **优点**：**计算效率极高**。MVSplat 能实现 20+ FPS 的推理速度，且深度估计通常比纯 Attention 方法更鲁棒，生成的几何结构更平整。
- **缺点**：深度平面的采样范围需要预设，对极近或极远的物体可能覆盖不足。

### 2.3 高斯属性解码器 (The Gaussian Head)

当网络通过上述机制确定了每个像素对应的特征向量 $f_{uv}$ 和深度 $d_{uv}$ 后，解码器头（prediction heads）将回归高斯的其他属性。

*   **不透明度 ($\alpha$)**：通常经过 Sigmoid 激活。网络会自动学会将天空或无效区域的 $\alpha$ 预测为 0。
*   **缩放 ($S$)**：通常预测 log-scale。
*   **旋转 ($R$)**：预测四元数（Quaternion），并进行归一化。
    *   *Rule-of-Thumb*: 前馈网络很难直接预测绝对旋转。通常预测相对于相机光轴的偏转，或者相对于射线方向的旋转。
*   **颜色/球谐 ($SH$)**：直接预测 RGB 值或低阶 SH 系数。部分方法（如 MVSplat）为了泛化性，不直接预测颜色，而是预测颜色特征，再通过像 NVS 那样的渲染网络解码，但这在 3DGS 框架下通常直接回归 SH 系数以便于后续快速光栅化。

---

## 3. 非配准数据与 Wild Data：AnySplat 的崛起

上一节讨论的 pixelSplat/MVSplat 依然假设输入包含了**完美的相机位姿**。但在真实世界的“随手拍”视频中，我们往往没有 Pose。传统的解决办法是先跑 COLMAP，但这正是我们想避免的。

2024-2025 年涌现的 **AnySplat**、**DL3DV-Recon** 等工作，引入了**几何基础模型（Geometric Foundation Models）** 来解决这一难题。

### 3.1 几何基座：DUSt3R / MASt3R
**DUSt3R** (Dense Unconstrained Stereo 3D Reconstruction) 是一个颠覆性的工作。它不进行传统的特征匹配+BA，而是将 3D 重建视为**全图点云回归（Pointmap Regression）**任务。
- 给定两张图，DUSt3R 直接输出两张对应的 3D 点图（Pointmap）和置信度图。
- 它可以直接对齐多张图片，得到一个粗糙但全局一致的相机位姿和点云，且**不需要相机内参**。

### 3.2 AnySplat 的工作流
AnySplat 巧妙地将 DUSt3R 的输出作为 3DGS 的“热启动”信号，取代了 COLMAP。

1.  **粗几何初始化**：使用 DUSt3R 处理输入的无序图片，得到粗糙的相机位姿 $P_{init}$ 和稀疏点云 $XYZ_{init}$。
2.  **高斯初始化**：在 DUSt3R 生成的点云位置生成高斯。
3.  **联合优化 (Joint Optimization)**：
    由于 DUSt3R 的位姿并不完美（存在尺度漂移或非刚性变形），AnySplat 引入了一个可微分的 Pose 优化模块。
    $$
    \mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_1 \mathcal{L}_{\text{depth\_consistency}} + \lambda_2 \mathcal{L}_{\text{semantic}}
    $$
    在训练高斯参数的同时，**梯度会反向传播到相机参数 $P$** 上，对其进行微调（Fine-tuning）。

### 3.3 语义感知的稳健性设计
在野外数据（Wild Data）中，移动的行人、车辆是重建的噩梦。AnySplat 利用视觉大模型（如 DINOv2, SAM）生成语义 Mask。
- **Masking Strategy**：对于语义识别为“人”、“车”的区域，降低光度损失的权重，避免高斯试图去拟合移动物体而在空中产生“幽灵”伪影。
- **Feature Metric Loss**：除了 RGB Loss，还计算 DINO 特征的距离。这在弱纹理区域（如纯色墙面）提供了额外的监督信号，防止高斯漂移。

---

## 4. 泛化能力与微调 (Generalization vs. Fine-tuning)

工程实践中，我们通常不会只用前馈网络的输出作为最终结果。前馈网络受限于训练数据的分辨率，生成的纹理往往比较模糊。

**最佳实践流程（Hybrid Pipeline）**：
1.  **Step 1: Inference (毫秒级)**
    使用 MVSplat/AnySplat 预训练模型，输入图像，直接预测出一套 3DGS 参数。
    *此时：几何结构基本正确，视角切换流畅，但纹理细节稍差。*
2.  **Step 2: Fast Fine-tuning (秒级)**
    将 Step 1 得到的高斯作为**初始值**，在当前图像上运行 300~500 步标准的 3DGS 优化。
    *此时：高频纹理细节被恢复，同时也修正了前馈网络的一些几何误差。*

这种 "Inference + Fine-tuning" 的策略，比从随机/SfM点云开始训练要快 10-50 倍，且质量更高。

---

## 5. 本章小结

1.  **前馈即回归**：通用 3DGS 将重建视为从图像到高斯参数的回归问题，核心在于利用 CNN/ViT 提取特征并显式构建多视图几何约束。
2.  **Cost Volume 优于 Attention**：在 2025 年的语境下，基于 Cost Volume (MVSplat) 的方法在速度和几何质量上通常优于纯 Epipolar Attention (pixelSplat) 方法。
3.  **Foundation Model 取代 SFM**：AnySplat 证明了利用 DUSt3R 等模型进行粗几何初始化，再结合 3DGS 的联合优化，可以彻底抛弃 COLMAP，实现无 Pose 视频的鲁棒重建。
4.  **混合策略**：前馈网络提供极佳的初始化，微调（Fine-tuning）提供极致的细节。二者结合是目前工程落地的最优解。

---

## 6. 常见陷阱与错误 (Gotchas)

### 6.1 坐标系地狱 (Coordinate System Hell)
在前馈网络中，坐标系的定义至关重要。
*   **陷阱**：直接预测世坐标系（World Space）的高斯是不可行的，因为网络无法感知绝对位置。
*   **解决**：通常以**参考视角（Reference View）**的相机坐标系为基准。在多视图融合时，需要精确处理 $T_{src \to ref}$ 的变换。如果外参矩阵弄反了（左乘右乘、逆矩阵），网络完全无法收敛。
*   **AnySplat 特有**：DUSt3R 输出的坐标系是归一化的相对坐标，与真实的物理尺度（米）没有关系。在与 AR 系统结合时，需要进行 Sim3 对齐。

### 6.2 显存峰值 (Peak VRAM Usage)
构建 Cost Volume 非常消耗显存。
*   **现象**：在 4090 上跑 pixelSplat 只有 1FPS，或者直接 OOM。
*   **技巧**：
    1.  **梯度检查点 (Gradient Checkpointing)**：牺牲计算换显存。
    2.  **降低 Cost Volume 分辨率**：特征图可以是原图的 1/4 或 1/8，预测出低分辨率深度图后，再通过插值上采样去指导高斯生成。

### 6.3 泛化模型的“域偏差” (Domain Gap)
*   **现**：在 RealEstate10K（室内房产）上训练的模型，拿去跑户外街景，天空可能会变成一堵墙，或者远处的树木深度被拉得很近。
*   **原因**：深度先验过拟合了室内场景的尺度范围。
*   **解决**：
    1.  使用在混合数据集（Objaverse + ScanNet + DL3DV）上训练的 Checkpoint。
    2.  **Test-time Adaptation**：在推理时，利用 AnySplat 的自监督 Loss 微调网络的 BatchNorm 层或 Pose。

### 6.4 动态物体的重影
*   **陷阱**：在无 Pose 优化时，如果画面中有一辆车开过，联合优化很容易把相机位姿“带偏”，以为是相机在动而不是车在动。
*   **Rule-of-Thumb**：对于单目/非同步视频，必须强制加上语义 Mask 忽略动态物体，否则几何必崩。

---

> **下一章预告**：解决了“单次采集、快速生成”的问题后，我们将在 **第 11 章** 面对更复杂的时空挑战——**多遍历自动驾驶场景**。当我们在不同季节、不同气反复经过同一条街道，如何融合这些数据并解耦动态交通流？我们将深入探讨 MTGS 等大规模建图方案。
