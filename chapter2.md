# 第 2 章 · 数学基础与渲染管线：3DGS 的心脏

> **摘要**
> 本章是理解 3D Gaussian Splatting (3DGS) 的基石。我们将剥开 PyTorch 代码的表象，深入底层数学原理。不同于 NeRF 的隐式体积渲染（Ray-marching），3DGS 是一套基于**显式各向异性高斯球**的**光栅化**系统。
>
> 本章重点覆盖：
> 1.  **协方差代数**：如何用四元数和缩放因子构建合法的 3D 椭球。
> 2.  **EWA Splatting**：从 3D 分布到 2D 图像平面的投影数学推导（Jacobian 近似）。
> 3.  **Tile-based Rasterizer**：高性能并行渲染管线的详细拆解（排序、混合、早停）。
> 4.  **梯度流与密度控制**：如何通过视图空间的梯度反推 3D 结构的生长与消亡。

---

## 2.1 3D 高斯：场景的“软原子”

在传统的显式几何中，我们习惯用三角形网格（Mesh）或点云（Point Cloud）。Mesh 拓扑难以改变，点云缺乏连续性且存在空洞。3DGS 引入了“3D 高斯”作为基本单元，你可以将其想象为一个边缘模糊的、半透明的椭球体。

### 2.1.1 为什么选择高斯函数？
高斯函数具有极其优秀的数学性质：
1.  **封闭性**：高斯函数的投影依然是高斯函数（Affine transform invariant）。这意味着我们不需要在 2D 屏幕上重新积分，直接算参数即可。
2.  **光滑可微**：$C^ \infty$ 连续，梯度可以传播到参数空间的任何地方，非常适合梯度下降优化。
3.  **稀疏性**：虽然高斯定义域是无穷大，但实际上 $3\sigma$ 之外的值极小，可以安全裁剪，利于光栅化性能。

### 2.1.2 协方差矩阵的参数化 (The Covariance Math)
一个中心在 $\mu$ 的 3D 高斯分布由下式定义：
$$
G(x) = \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)
$$
核心在于协方差矩阵 $\Sigma$。它控制了椭球的大小、旋转和形状。
然而，**我们不能直接优化 $\Sigma$ 的 9 个元素**。
*   **原因**：$\Sigma$ 必须是**半正定矩阵 (Positive Semi-Definite)**。如果梯度下降导致 $\Sigma$ 变成了非正定矩阵，这就不是一个椭球了（可能变成双曲面），物理意义崩塌，渲染程序报错。

**解决方案：分解为旋转与缩放**
类似于特征值分解，我们将 $\Sigma$ 构造为：
$$
\Sigma = R S S^T R^T
$$
*   **缩放矩阵 $S$**：对角矩阵 $S = \text{diag}(s_x, s_y, s_z)$。
    *   *实现细节*：实际优化的是 $\hat{s} \in \mathbb{R}^3$，使用 $s = \exp(\hat{s})$ 激活。这保证了缩放因子永远非负。
*   **旋转矩阵 $R$**：由四元数 $q = (w, x, y, z)$ 转换而来。
    *   *实现细节*：优化 $q$，每次使用前做归一化 $q_{norm} = \frac{q}{\|q\|}$ 得到合法的旋转矩阵。

> **💡 Rule of Thumb (缩放陷阱)**
> 在初化时，$S$ 的值通常设为较小的值（如最近邻点距离的平均值）。如果 $S$ 过大，高斯会重叠严重，导致优化初期“模糊一团”；如果 $S$ 过小，场景会出现大量“针尖”空洞，梯度难以传播。

---

## 2.2 外观建模：球谐函数 (Spherical Harmonics)

NeRF 使用 MLP 来拟合 View-Dependent Color，而 3DGS 回归了传统图形学的 **球谐函数 (SH)**。

### 2.2.1 什么是 SH？
SH 是定义在球面上的正交基函数组，类似于 1D 信号处理中的傅里叶级数，SH 是 2D 球面信号的频域分解。
$$
C(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_{lm}(\theta, \phi)
$$

*   **Degree 0 (1系数)**：漫反射 (Diffuse)。不论从哪个角度看颜色都一样。
*   **Degree 1 (3系数)**：简单的方向性光照。
*   **Degree 2 (5系数)** & **Degree 3 (7系数)**：高频反射，如金属高光、各向异性材质。

### 2.2.2 优化策略
3DGS 每个高斯携带 $k$ 个 SH 系数向量（对于 RGB 三通道，总系数为 $3 \times (L+1)^2$）。
*   **L=3** 时，每个高斯需要 $3 \times 16 = 48$ 个浮点数来存颜色。这对显存是巨大压力。
*   **渐进式训练**：初始阶段通常只训练 DC 项（Degree 0），待几何稳定后（例如 1000 iters 后），逐步解锁 Degree 1, 2, 3。

> **⚠️ Gotcha (SH 伪影)**
> 如果过早引入高阶 SH，模型会倾向于把几何误差“藏”在颜色里（View-dependent overfitting）。表现为：静态图片看着还行，一动起来高光乱闪，像迪斯科球。

---

## 2.3 投影机制：EWA Splatting (3D $\to$ 2D)

这是 3DGS 区别于点云渲染最“数学”的部分。我们需要计算 3D 椭球投影到屏幕上是什么样子的 **2D 椭圆**。

### 2.3.1 线性化投影
给定视图变换 $W$ (World to Camera) 和投影变换 $P$ (Camera to Screen)。
透视投影是非线性的，但高斯变换要求线性。因此，我们在高斯中心 $\mu$ 处对投影函数进行**局部泰勒展开**（Local Affine Approximation）。

令 $J$ 为投影变换的雅可比矩阵（Jacobian Matrix），它描述了相机空间微小位移如何映射到屏幕像素空间：
$$
J = \begin{bmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} & \frac{\partial u}{\partial z} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial z}
\end{bmatrix}
$$
其中 $(u, v)$ 是像素坐标，$(x, y, z)$ 是相机坐标系下的点。

### 2.3.2 2D 协方差公式
基于 EWA (Elliptical Weighted Average) 理论，投影后的 2D 协方差 $\Sigma'$ 为：
$$
\Sigma' = J W \Sigma W^T J^T
$$
这里其实省略了一步：为了抗锯齿，通常会加上一个低通滤波器项（如 $0.3 I$），防止当 3D 高斯远小于 1 个像素时产生走样（Aliasing）。

$$
\Sigma'_{\text{final}} = \Sigma' + \sigma_{\text{blur}}^2 I
$$

计算出 $\Sigma'$ 后，我们就得到了 2D 屏幕上的一个高斯分布。

> **ASCII 图解：Jacobian 的作用**
>
> ```text
>      ^ Camera Plane
>      |      /
>      |     /  Projection Rays
>      |    /
>   [3D Ellipsoid]  <-- Jacobian linearizes the rays here
>         \
>          \
>           \
>         [2D Ellipse] on Image Plane
> ```
> *如果高斯球离相机很近，视角很大，线性近似会产生误差。但在 3DGS 中，由于高斯通常很小，这个近似非常精确。*

---

## 2.4 高性能光栅化管线 (The Rasterizer)

3DGS 的速度核心在于：**它不发射光线（No Ray-marching），它只做排序（Sorting）。**

### 2.4.1 预处理 (Pre-pass)
在每一帧渲染开始时，CUDA Kernel 会并行处理所有高斯：
1.  **Frustum Culling**：检查高斯是否在视锥体内。
    *   *技巧*：不仅检查中心，还要检查 $3\sigma$ 边界框（AABB）。
2.  **Projection**：计算 2D 坐标、深度 $d$ 和 2D 协方差 $\Sigma'$。
    *   *数值稳定性*：如果 depth 极小（靠近 Near Plane），Jacobian 会爆炸。通常需要在这里做截断保护。
3.  **Tile Key 生**：将屏幕分为 $16 \times 16$ 的 Tiles。计算每个高斯覆盖了哪些 Tile。

### 2.4.2 关键排序 (Radix Sort)
这是管线中最“重”的一步。
*   我们生成一个列表，每个元素是 `(Key, Value)`。
*   **Key** = `(Tile ID << 32) | (Depth_Bit_Representation)`。
*   通过对 Key 进行 **GPU Radix Sort**，我们实现了两个目标：
    1.  所有属于同一个 Tile 的高斯在内存中连续存放。
    2.  同一个 Tile 内的高斯按深度（从近到远）排列。

### 2.4.3 Tile 混合 (Blending)
每个 Tile 启动一个 Thread Block（例如 256 线程）进行像素着色：
1.  线程协作将该 Tile 涉及到的高斯数据（位置、颜色、透明度）加载到 **Shared Memory**。
2.  每个像素线程遍历排序后的高斯列表，执行 Front-to-Back 混合：

$$
C_{\text{final}} = \sum_{i=1}^{N} c_i \alpha_i T_i, \quad \text{where } T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

*   $\alpha_i$：第 $i$ 个高斯在当前像素的贡献（由 2D 高斯值 $\times$ 自身不透明度得到）。
*   $T_i$：透射率（Transmittance），即光线到达第 $i$ 个高斯时还剩多少。

### 2.4.4 早停机制 (Early Termination)
这是性能优化的关键：
$$
\text{if } T_i < 0.0001 \quad \text{break;}
$$
当一个像素已经累积了 99.99% 的不透明度（变得不透光了），其后的所有高斯（即使有几千个）都可以直接跳过计算。对于遮挡严重的场景，这能带来数倍的加速。

---

## 2.5 自适应密度控制 (Adaptive Density Control)

3DGS 并不是用固定数量的点去拟合，而是一个动态的**生死系统**。

### 2.5.1 梯度作为生长信号
在反向传播时，我们关注**视图空间位置的梯度** $\nabla_{\mu_{2D}} \mathcal{L}$。
*   **直觉**：如果一个高斯试图疯狂移动它的 2D 位置来降低 Loss，说明这个位置的几何描述是错误的（要么是缺东西，要么是太大了）。
*   我们统计每个高斯在一定迭代周期内（如每 100 iters）的平均梯度幅值。

### 2.5.2 克隆 (Clone) 与 分裂 (Split)
若某个高斯的平均梯度大于阈值 $\tau_{pos}$（如 0.0002）：
1.  **小高斯 $\to$ 克隆**：如果高斯本身很小（方差小于阈值），说明这里是高频细节（如树叶、纹理），需要更多点来填充。$\to$ **复制一份**。
2.  **大高斯 $\to$ 分裂**：如果高斯很大，说明它试图用一个椭球覆盖形状复杂的区域（欠拟合）。$\to$ **分裂成两个小高斯**，位置略微偏移，方差缩小为原来的 $\phi$ 倍（通常 1.6）。

### 2.5.3 剪枝 (Prune)
定期清理“垃圾”高斯：
1.  **几乎透明**：Opacity $< 0.005$。对画面没贡献，浪费显存。
2.  **过大膨胀**：世界空间尺度过大，通常是优化失败的伪影（Floaters）。

> **💡 Rule of Thumb (Opacity Reset)**
> 3DGS 有一个著名的技巧：**Opacity Reset**。每隔 3000 iters，强制将所有高斯的 Opacity 设为 0.01（或是接近 0 的值）。
> *作用*：这迫使那些“侥幸”存活的、非必要的模糊高斯被剪枝掉，倒逼系统重新从核心几何生长，极大提升了重建锐度。

---

## 2.6 本章小结

1.  **表示层**：3DGS 使用分解的协方差矩阵 ($\Sigma = RSS^T R^T$) 保证几何合法性，用 SH 系数拟合视角颜色。
2.  **投影层**：利用雅可比矩阵局部线性化，将 3D Splat 闭式投影为 2D Splat。
3.  **计算层**：抛弃 Ray-marching，拥抱 **Sorting + Rasterization**。通过 Tile-based 渲染和早停机制压榨 GPU 性能。
4.  **优化层**：不仅优化参数，还利用梯度流动态**增删改**高斯数量，实现了结构与外观的协同进化。

---

## ⚠️ 常见陷阱与错误 (Gotchas)

1.  **World vs Camera 坐标系混乱**
    *   GSplat 库和官方实现通常假设 View Matrix 是 $T_{world \to camera}$。如果你传入的是 Pose ($T_{camera \to world}$)，你会发现场景虽然重建出来了，但是是在做“反向运动”，或者根本无法敛。
    *   *调试*：始终检查 $\det(R) = 1$ 且平移向量是否符合物理直觉。

2.  **学习率敏感性**
    *   **位置学习率**：如果设得太高，高斯会到处乱飞，无法稳定成形；太低则收敛极慢。通常需要指数衰减调度（Exponential Decay）。
    *   **Scale 学习率**：Scale 必须学习得比 Position 慢，否则高斯倾向于迅速膨胀成大球来降低 Loss，而不是移动到正确位置。

3.  **近平面裁剪 (Near Plane Clipping)**
    *   当相机穿过物体内部时，部分高斯中心在相机背后 $(z<0)$，部分在前面。
    *   简单的投影公式在 $z \to 0$ 时会除以零。必须在 Shader 中实现严格的 Guard Band 剔除，否则会出现占据整个屏幕的单一颜色闪烁。

4.  **2025 新视角的陷阱：结构化退化**
    *   原始 3DGS 在无纹理区域（白墙、天空）容易产生极其细长的高斯（Needle-like shapes）。这在合成新视角时没问题，但如果你想把这些高斯导出为 Mesh 或用于物理碰撞，由于几何极度扭曲，基本不可用。
    *   *解决*：2025 年的改进版（如 Regularized GS, 2DGS）通常引入了**各向同性正则化**项 $\mathcal{L}_{iso} = ||s_x - s_y||$，强迫高斯保持相对扁平或球状。
