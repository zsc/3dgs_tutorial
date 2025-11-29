# 第 7 章 · 3DGS + SLAM：GS-SLAM, RTG-SLAM 与在线重建

> **本章目标**：深入理解基于 3D Gaussian Splatting 的 SLAM 系统设计。从 Tracking 与 Mapping 的数学对偶性出发，剖析 GS-SLAM、SplaTAM、RTG-SLAM 等 SOTA 架构。重点掌握在线化过程中的**关键帧管理**、**动态物体剔除**、**回环检测与地图融合**策略，以及如何处理 RGB-D 与单目两种输入模式。

---

## 7.1 引言：稠密 SLAM 的“不可能三角”

在 3DGS 出现之前，视觉 SLAM 领域长期面临一个“不可能三角”的权衡：
1.  **稀疏 vs 稠密**：ORB-SLAM 等特征点法速度快、精度高，但只能输出稀疏点云，无法直接用于导航避障或 AR 遮挡。
2.  **实时性 vs 质量**：KinectFusion、ElasticFusion 等稠密方法受限于 TSDF 体素分辨率，大场景下显存和计算量呈立方级增长。
3.  **可塑性 vs 遗忘**：基于 NeRF 的 SLAM（如 iMAP, NICE-SLAM）解决了连续性问题，但训练极慢，且面临“灾难性遗忘”（Catastrophic Forgetting）——优化新房间时，旧房间的几何可能崩坏。

**3D Gaussian Splatting 的入局改变了这一切**。它用离散的“高斯球”代替了神经网络权重或体素网格，同时实现了：
*   **>100 FPS 的渲染速度**：支持高频 Tracking。
*   **局部更新特性**：修改地图的某个角落不会影响全局，天然适合增量式 SLAM。
*   **显式几何**：可直接转化为点云或 Mesh，无需昂贵的 Marching Cubes。

---

## 7.2 3DGS-SLAM 的核心架构与数学原理

大多数 3DGS SLAM 系统都遵循经典的 **前端（Tracking）** + **后端（Mapping）** 并行架构。

### 7.2.1 系统数据流 (Pipeline)

```text
Incoming Stream (RGB + Depth + Pose_init)
       |
       v
+-----------------------+           +------------------------+
|   Frontend: Tracking  | <-------> |    Global Map (G)      |
|  Min(Photo_Loss) -> T |           |   List[3D Gaussians]   |
+-----------------------+           +-----------+------------+
       |                                        ^
       | (Is Keyframe?)                         |
       v                                        | (Update)
+-----------------------+                       |
|   Backend: Mapping    | ----------------------+
|  Add / Prune / Clone  |  Optimization Window
|  Refine Gaussians     |
+-----------------------+
```

### 7.2.2 跟踪（Tracking）：逆向渲染问题

在 Tracking 阶段，我们冻结高斯地图参数 $G$，求解当前帧的相机位姿 $T \in SE(3)$。
位姿通常由李代数 $\xi \in \mathfrak{se}(3)$ 参数化，通过指数映射 $T = \exp(\xi)$ 作用于相机。

目标函数是最小化光度残差与几何残差：
$$
E(\xi) = \sum_{p \in \Omega} \lambda_c \| I_{gt}(p) - I_{render}(p, \xi, G) \|_1 + \lambda_d \| D_{gt}(p) - D_{render}(p, \xi, G) \|_1
$$

**技术难点：梯度流向**
在 Vanilla 3DGS 中，梯度是对高斯参数 $\mu, \Sigma, c, \alpha$ 求导。但在 SLAM Tracking 中，我们需要对相机外参求导。
$$
\frac{\partial E}{\partial \xi} = \frac{\partial E}{\partial I} \cdot \frac{\partial I}{\partial \mu_{2D}} \cdot \frac{\partial \mu_{2D}}{\partial \mu_{3D}} \cdot \frac{\partial \mu_{3D}}{\partial T} \cdot \frac{\partial T}{\partial \xi}
$$
其中关键项是 $\frac{\partial \mu_{3D}}{\partial T}$，即 3D 高斯中心点随相机位姿变化的雅可比矩阵。

> **Rule of Thumb (跟踪鲁棒性)**：
> 纯光度误差是一个高度非凸（Non-convex）的函数，容易陷入局部极小值。
> 1. **恒速模型 (Constant Velocity Model)**：用上一帧的速度预测当前帧位姿作为初值，至关重要。
> 2. **图像金字塔 (Pyramid)**：先在 1/4 分辨率下优化位姿，再在全分辨率下微调，可以扩大收敛域，抵抗大运动。

### 7.2.3 建图（Mapping）：增量式生长与维护

Mapping 线程负责更新 $G$。与离线 3DGS 不同，SLAM 不能每次都从头训练，必须采用**滑窗优化 (Sliding Window Optimization)**。

1.  **关键帧选择 (Keyframe Selection)**：当当前帧与上一关键帧的视差（IoU 或 相对位姿）超过阈值时，将其插入关键帧数据库。
2.  **高斯初始化 (Initialization)**：
    *   利用 RGB-D 的深度图，将像素反投影（Unproject）到 3D 空间，在空白区域生成新的高斯球。
    *   **深度不确定性**：对于深度缺失或噪声大的区域（如物体边缘、反光面），初始化的方差应设得较大。
3.  **致密化与剪枝 (Densification & Pruning)**：
    *   **Clone/Split**：策略与原始 3DGS 类似，基于位置梯度。
    *   **Pruning**：SLAM 中极其重要。必须及时剔除那些被判定为“遮挡”或“错误深度”的高斯，否则地图会因噪点积累而产生严重的“雾化”现象。

---

## 7.3 主流方案深度剖析

### 7.3.1 GS-SLAM 与 SplaTAM：RGB-D 的标准范式

这两项工作（CVPR 2024 等）奠定了 RGB-D GS-SLAM 的基础。

*   **SplaTAM 的“自由空间”约束**：
    SplaTAM 引入了一个显式的先验：如果深度传感器告诉我们前方 2 米是空的，那么相机到 2 米之间的光线穿透率（Transmittance）应该接近 1。
    $$ \mathcal{L}_{empty} = \sum_{samples} \alpha_i \approx 0 $$
    这有效地消除了相机前方的漂浮伪影（Floaters）。

*   **GS-SLAM 的自适应扩张**：
    GS-SLAM 提出了一种基于几何误差的扩张策略。当渲染深度与观测深度不一致时，说明该处缺乏几何描述，系统会强制在该区域生成新的高斯，而不仅仅依赖颜色梯度。

### 7.3.2 单目 GS-SLAM：尺度与深度的博弈

单目（Monocular）SLAM 没有真实的深度输入，直接运行 3DGS 会导致尺度模糊和深度坍塌。

*   **MonoGS / Gaussian-SLAM**：通常引入单目深度估计网络（如 Depth Anything, ZoeDepth）作为伪真值（Pseudo-GT）。
*   **尺度对齐**：由于单目深度网络的尺度是相对的，系统需要实时计算一个缩放因子 $s$ 和偏移 $b$，将预测深度 $D_{pred}$ 对齐到当前的地图尺度：$D_{map} \approx s \cdot D_{pred} + b$。

### 7.3.3 RTG-SLAM 与紧凑化

为了在大场景下保持实时性，**RTG-SLAM** 限制了高斯球的总量。
*   它不追求完美的纹理复现，而是追求几何结构的准确性。
*   引入了**结构化高斯**的思想，对于平坦区域（墙面、地面），强制将高斯球压扁（Scaling 的一个轴趋近于 0），用更少的高斯覆盖更大的面积。

---

## 7.4 动态环境下的 SLAM

真实场景中充满了行人、宠物和移动的家具。动态物体会污染 SLAM 的地图（留下残影）并干扰 Tracking（导致定位漂移）。

### 7.4.1 动态检测策略

如何在没有标注的情况下发现动态物体？

1.  **光度/几何残差法**：
    在 Tracking 收敛后，计算残差图 $R = |I_{gt} - I_{render}|$。如果是静态场景，残差应服从高斯噪声分布。如果某块区域残差异常大（Outliers），则判定为动态。
    
2.  **语义辅助法**：
    利用 YOLO 或 Mask2Former，检测“人”、“车”、“猫”等潜在动态类别。

### 7.4.2 动静分离架构

处理动态物体的终极方案是**双流（Two-stream）设计**：

*   **Static Map**：仅包含背景高斯。用于相机 Tracking。
*   **Dynamic Map**：包含移动物体的高斯。
*   **Pipeline**：
    1. 输入帧 $I_t$。
    2. 使用上一帧的运动模型预测动态物体的 Mask。
    3. 仅使用 Static Mask 区域的像素进行 Camera Pose Optimization。
    4. 剩余像素用于更新 Dynamic Map（或直接丢弃，如果不关心动态重建）。

> **Gotcha**：即使使用了语义 Mask，物体的**阴影**投射在静态地面上，也会产生光度误差。高级的系需要将阴影也建模为外观变化，或者使用对光照不敏感的特征（如 Feature Metric SLAM）进行跟踪。

---

## 7.5 进阶话题：回环检测与子图融合

传统的 3DGS 是一张全局的大图。但在 SLAM 中，长时间运行会产生累积漂移（Drift）。当回到原点时，如果不做处理，会出现“双重影”。

### 7.5.1 子图（Sub-map）策略

为了解决漂移，现代 3DGS-SLAM（如 2025 年的 Multi-Map Approaches）倾向于使用子图：

1.  将整个轨迹切分为若干个局部子图（Local Sub-maps）。
2.  每个子图有自己的局部坐标系和一组高斯球。
3.  **回环检测 (Loop Closure)**：使用 BoW (Bag of Words) 或 NetVLAD 检索历史关键帧。
4.  **位姿图优化 (Pose Graph Optimization, PGO)**：当检测到回环，计算相对位姿变换，优化全局的关键帧位姿。

### 7.5.2 地图融合 (Map Merging)

PGO 优化了关键帧位姿后，如何修正高斯地图？
*   **刚性变换**：如果子图小，可以直接将整个子图的高斯球坐标乘以新的变换矩阵 $T_{optimized}$。
*   **非刚性变形 (Non-rigid Deformation)**：更精细的方法是根据控制点（Control Points）的位移，对空间中的高斯球场进行插值变形（Warping），使闭环处的接缝平滑融合。

---

## 7.6 本章小结

*   **范式转变**：3DGS-SLAM 将三维重建从“离线、全量”推向了“在线、增量”。
*   **性能权衡**：利用高斯球的快速渲染实现实时跟踪，但需要精细的显存管理（剪枝）以防止内存爆炸。
*   **RGB-D 是主流**：目前最鲁棒的系统仍依赖深度相机。单目系统需要借助深度估计网络。
*   **动态与回环**：这是从 Demo 走向实用的关键。动静分离解决干扰，子图融合解决漂移。

---

## 7.7 常见陷阱与调试技巧 (Gotchas)

### 1. 深度图的预处理陷阱
*   **问题**：商业深度相机（Realsense/Kinect）在物体边缘会有严重的“飞点”或噪声，且对黑色物体、反光面失效（深度为 0）。
*   **后果**：如果在这些位置初始化高斯，会在空中产生大量垃圾伪影，遮挡视线。
*   **技巧**：必须对深度图进行腐蚀（Erosion）操作和双边滤波（Bilateral Filter），并设置严格的有效深度阈值（如 0.2m - 5.0m），丢弃过远或无效的深度值。

### 2. 学习率的敏感性
*   **问题**：Tracking 经常跟丢（Loss 突然爆炸）。
*   **原因**：Tracking 的学习率（针对 Pose）和 Mapping 的学习率（针对 Gaussians）如果不匹配，会导致“地图追着相机跑”或“相机剧烈震荡”。
*   **技巧**：Tracking 的 LR 通常需要随迭代次数衰减。且在 Mapping 阶段，新添加的高斯的 Opacity 应该从低值开始 warm-up，避免突然遮挡已有的正确几何。

### 3. "Z-fighting" 与地图重叠
*   **问题**：在回环处，新旧地图重叠，渲染时出现闪烁。
*   **原因**：两个高斯球在同一位置，深极度接近，Rasterizer 排序不稳定。
*   **技巧**：这通常需要后端融合（Merge）逻辑，检测空间上重叠度极高的高斯球（位置近、协方差相似、颜色相似），将其合并为一个，减少冗余。

### 4. 显存 OOM (Out Of Memory)
*   **调试**：不要在此刻相信 PyTorch 的自动显存管理。在 SLAM 循环中，显存只会增加。
*   **技巧**：
    *   定期调用 `torch.cuda.empty_cache()`。
    *   实现严格的**视锥体剔除**：只将当前相机视野内的高斯加载到优化器中，其他高斯 "冻结" 或卸载到 CPU 内存。
