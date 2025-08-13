# 第 11 章：多维 Stride DMA 利用

本章深入探讨 AI 编译器如何高效利用现代硬件的多维 DMA（Direct Memory Access）能力，实现张量数据的高效传输。我们将分析 stride 访问模式、多维描述符设计、以及如何将高层张量操作映射到底层 DMA 指令。这些技术对于优化内存带宽利用率和减少数据传输开销至关重要，特别是在处理大规模神经网络和实时推理场景中。

## 11.1 多维 DMA 描述符设计

### 11.1.1 传统 DMA 的局限性

传统的一维 DMA 只能处理连续内存块的传输，对于多维张量操作存在严重限制：

```
传统 DMA 传输模式：
Source: [A0, A1, A2, ..., An-1]  →  Dest: [B0, B1, B2, ..., Bn-1]
```

当需要传输张量的子块或进行维度变换时，需要多次 DMA 操作：

```
2D 张量子块传输（使用 1D DMA）：
Tensor A (4×6):          提取 2×3 子块：
[a00 a01 a02 a03 a04 a05]   需要 2 次 DMA：
[a10 a11 a12 a13 a14 a15]   1. 传输 [a11 a12 a13]
[a20 a21 a22 a23 a24 a25]   2. 传输 [a21 a22 a23]
[a30 a31 a32 a33 a34 a35]
```

### 11.1.2 多维 DMA 描述符结构

现代 AI 加速器的多维 DMA 描述符包含以下关键字段：

$$
\text{DMA}_{\text{desc}} = \{
\begin{aligned}
&\text{src\_addr}, \text{dst\_addr}, \\
&\text{dim\_count}, \\
&\text{size}[d], \text{src\_stride}[d], \text{dst\_stride}[d] \quad \forall d \in [0, \text{dim\_count})
\end{aligned}
\}
$$

其中：
- $\text{size}[d]$：第 $d$ 维的元素数量
- $\text{src\_stride}[d]$：源地址第 $d$ 维的步长（字节）
- $\text{dst\_stride}[d]$：目标地址第 $d$ 维的步长（字节）

### 11.1.3 地址计算公式

对于 $n$ 维 DMA，源地址和目标地址的计算公式为：

$$
\text{addr}_{\text{src}}(i_0, i_1, ..., i_{n-1}) = \text{src\_base} + \sum_{d=0}^{n-1} i_d \cdot \text{src\_stride}[d]
$$

$$
\text{addr}_{\text{dst}}(i_0, i_1, ..., i_{n-1}) = \text{dst\_base} + \sum_{d=0}^{n-1} i_d \cdot \text{dst\_stride}[d]
$$

### 11.1.4 描述符优化策略

**层次化描述符设计**：

```
3D DMA 描述符层次：
Level 2 (Batch):  size=2, stride=1024
  │
  ├─ Level 1 (Row): size=4, stride=128
  │     │
  │     └─ Level 0 (Col): size=8, stride=4
```

**描述符压缩**：利用规则性减少存储开销：

$$
\text{Compressed\_Desc} = \begin{cases}
\text{Regular}: & \text{base, size, stride} \\
\text{Strided}: & \text{base, size}[2], \text{stride}[2] \\
\text{Irregular}: & \text{full descriptor}
\end{cases}
$$

## 11.2 Stride 访问模式分析

### 11.2.1 常见 Stride 模式分类

AI 工作负载中的典型 stride 访问模式：

1. **单位步长（Unit Stride）**：
   $$\text{stride} = \text{element\_size}$$
   
2. **固定步长（Fixed Stride）**：
   $$\text{stride} = k \cdot \text{element\_size}, \quad k \in \mathbb{N}^+$$
   
3. **分块步长（Block Stride）**：
   $$\text{stride} = \text{block\_size} \cdot \text{element\_size}$$
   
4. **嵌套步长（Nested Stride）**：
   $$\text{stride}_{\text{outer}} = n \cdot \text{stride}_{\text{inner}}$$

### 11.2.2 Stride 模式的性能影响

内存带宽利用率公式：

$$
\eta_{\text{bandwidth}} = \frac{\text{useful\_data\_transferred}}{\text{total\_bus\_transactions} \times \text{bus\_width}}
$$

对于 stride 访问：

$$
\eta_{\text{stride}} = \frac{\text{element\_size}}{\max(\text{element\_size}, \lceil\frac{\text{stride}}{\text{cache\_line\_size}}\rceil \times \text{cache\_line\_size})}
$$

### 11.2.3 Stride 冲突检测

检测不同 stride 模式间的冲突：

$$
\text{Conflict}(s_1, s_2, n) = \begin{cases}
\text{True} & \text{if } \gcd(s_1, s_2) > n \\
\text{False} & \text{otherwise}
\end{cases}
$$

其中 $n$ 是 bank 数量或缓存路数。

### 11.2.4 Stride 优化变换

**Stride 最小化变换**：

给定张量访问模式 $A[i \cdot s_1 + j \cdot s_2]$，寻找变换 $T$ 使得：

$$
\min_{T} \sum_{d} |s'_d - 1| \quad \text{where } s' = T(s)
$$

**Stride 对齐优化**：

$$
s_{\text{aligned}} = \lceil \frac{s}{\text{align\_size}} \rceil \times \text{align\_size}
$$

## 11.3 张量切片与 DMA 映射

### 11.3.1 张量切片表示

张量切片操作的数学表示：

$$
T_{\text{slice}} = T[i_0:j_0:s_0, i_1:j_1:s_1, ..., i_{n-1}:j_{n-1}:s_{n-1}]
$$

其中 $i_k$ 是起始索引，$j_k$ 是结束索引，$s_k$ 是步长。

对应的 DMA 描述符生成：

$$
\text{DMA}_{\text{slice}} = \{
\begin{aligned}
&\text{src\_addr} = \text{base} + \sum_{k} i_k \cdot \prod_{l>k} \text{dim}_l \cdot \text{elem\_size} \\
&\text{size}[k] = \lceil \frac{j_k - i_k}{s_k} \rceil \\
&\text{stride}[k] = s_k \cdot \prod_{l>k} \text{dim}_l \cdot \text{elem\_size}
\end{aligned}
\}
$$

### 11.3.2 切片合并优化

**相邻切片检测**：

两个切片 $S_1$ 和 $S_2$ 可合并的条件：

$$
\text{Mergeable}(S_1, S_2) = \begin{cases}
\text{True} & \text{if } \exists d: S_1.\text{end}[d] = S_2.\text{start}[d] \land \\
& \quad \forall k \neq d: S_1[k] = S_2[k] \\
\text{False} & \text{otherwise}
\end{cases}
$$

**切片分解策略**：

将大切片分解为硬件友好的小切片：

$$
T_{\text{large}} = \bigcup_{i=0}^{n-1} T_{\text{tile}_i}
$$

其中每个 $T_{\text{tile}_i}$ 满足：
$$
\text{size}(T_{\text{tile}_i}) \leq \text{DMA\_buffer\_size}
$$

### 11.3.3 张量重排与 DMA 映射

**转置操作的 DMA 实现**：

对于矩阵转置 $B = A^T$：

```
源张量 A (M×N):           目标张量 B (N×M):
stride_src[0] = N          stride_dst[0] = 1
stride_src[1] = 1          stride_dst[1] = M
```

DMA 描述符：
$$
\text{DMA}_{\text{transpose}} = \{
\begin{aligned}
&\text{size}[0] = M, \quad \text{size}[1] = N \\
&\text{src\_stride}[0] = N \cdot \text{elem\_size} \\
&\text{src\_stride}[1] = \text{elem\_size} \\
&\text{dst\_stride}[0] = \text{elem\_size} \\
&\text{dst\_stride}[1] = M \cdot \text{elem\_size}
\end{aligned}
\}
$$

### 11.3.4 Padding 与边界处理

处理非对齐边界的 DMA 策略：

$$
\text{Padded\_size}[d] = \lceil \frac{\text{original\_size}[d]}{\text{tile\_size}[d]} \rceil \times \text{tile\_size}[d]
$$

边界条件检查：
$$
\text{Valid\_transfer}(i, j) = \begin{cases}
\text{Data} & \text{if } i < \text{height} \land j < \text{width} \\
\text{Padding\_value} & \text{otherwise}
\end{cases}
$$

## 11.4 非连续内存传输优化

### 11.4.1 Gather/Scatter 操作

**Gather 操作**（从非连续地址收集数据）：

$$
\text{dst}[i] = \text{src}[\text{index}[i]], \quad i \in [0, n)
$$

DMA 优化策略：
1. 索引排序以提高局部性
2. 批量传输相邻元素
3. 使用间接寻址 DMA 模式

**Scatter 操作**（分散写入非连续地址）：

$$
\text{dst}[\text{index}[i]] = \text{src}[i], \quad i \in [0, n)
$$

### 11.4.2 稀疏张量传输

CSR 格式稀疏矩阵的 DMA 传输：

```
稀疏矩阵结构：
values:   [v0, v1, v2, ...]
col_idx:  [c0, c1, c2, ...]
row_ptr:  [r0, r1, r2, ...]
```

DMA 传输策略：
$$
\text{DMA}_{\text{sparse}} = \{
\begin{aligned}
&\text{Phase1}: \text{传输 row\_ptr 数组} \\
&\text{Phase2}: \text{批量传输每行的 values 和 col\_idx} \\
&\text{Phase3}: \text{重组为目标格式}
\end{aligned}
\}
$$

### 11.4.3 内存访问模式优化

**Bank 冲突避免**：

对于 $B$ 个 bank，stride $s$ 的访问，冲突概率：

$$
P_{\text{conflict}} = \begin{cases}
0 & \text{if } \gcd(s, B) = 1 \\
\frac{1}{\gcd(s, B)} & \text{otherwise}
\end{cases}
$$

**交错访问优化**：

$$
\text{addr}_{\text{interleaved}}(i) = \text{base} + (i \bmod B) \cdot \text{chunk\_size} + \lfloor \frac{i}{B} \rfloor
$$

### 11.4.4 预取与流水线

DMA 传输流水线深度优化：

$$
\text{Pipeline\_depth} = \max\left(\lceil \frac{\text{compute\_time}}{\text{transfer\_time}} \rceil, 2\right)
$$

双缓冲策略的时间节省：

$$
T_{\text{saved}} = \min(\text{compute\_time}, \text{transfer\_time}) \times (n-1)
$$

其中 $n$ 是总传输次数。

## 11.5 2D/3D DMA 引擎编程模型

### 11.5.1 2D DMA 编程抽象

2D DMA 的基本编程模型将矩形数据块作为基本传输单元：

```
2D 传输参数：
┌─────────────────────────┐
│ Width (W) × Height (H)  │
│ Src_pitch: 源行间距      │
│ Dst_pitch: 目标行间距    │
└─────────────────────────┘
```

数学表示：
$$
\text{DMA}_{2D} = \{W, H, \text{src\_pitch}, \text{dst\_pitch}, \text{src\_base}, \text{dst\_base}\}
$$

地址生成公式：
$$
\text{addr}(x, y) = \text{base} + y \cdot \text{pitch} + x \cdot \text{elem\_size}
$$

### 11.5.2 3D DMA 编程模型

3D DMA 扩展到立方体数据传输：

```
3D 传输层次：
Depth (D) ──┐
            ├── Height (H) ──┐
                            ├── Width (W)
```

完整的 3D DMA 描述符：
$$
\text{DMA}_{3D} = \{
\begin{aligned}
&\text{dim}: [W, H, D] \\
&\text{stride}: [\text{elem\_size}, \text{row\_pitch}, \text{slice\_pitch}] \\
&\text{offset}: [\text{x\_off}, \text{y\_off}, \text{z\_off}]
\end{aligned}
\}
$$

### 11.5.3 卷积操作的 DMA 映射

将卷积的滑窗操作映射到 2D DMA：

对于卷积核 $K$ 和输入 $I$：
$$
O[m,n] = \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} K[i,j] \cdot I[m \cdot s + i, n \cdot s + j]
$$

DMA 传输策略：
1. **Im2col 变换**：将每个滑窗展开为列
2. **Direct convolution**：直接传输滑窗数据

Im2col 的 DMA 描述：
$$
\text{DMA}_{\text{im2col}} = \{
\begin{aligned}
&\text{window\_size}: k_h \times k_w \\
&\text{stride}: s \\
&\text{num\_windows}: \lceil \frac{H-k_h}{s} \rceil \times \lceil \frac{W-k_w}{s} \rceil
\end{aligned}
\}
$$

### 11.5.4 分块矩阵乘法的 DMA 调度

对于矩阵乘法 $C = A \times B$，分块策略：

$$
C_{ij} = \sum_{k} A_{ik} \times B_{kj}
$$

DMA 调度序列：
```
for i in range(0, M, tile_m):
    for j in range(0, N, tile_n):
        for k in range(0, K, tile_k):
            DMA_2D(A[i:i+tile_m, k:k+tile_k] → Local_A)
            DMA_2D(B[k:k+tile_k, j:j+tile_n] → Local_B)
            Compute(Local_C += Local_A × Local_B)
        DMA_2D(Local_C → C[i:i+tile_m, j:j+tile_n])
```

优化的双缓冲 DMA 时序：
$$
T_{\text{total}} = T_{\text{init}} + (n-1) \cdot \max(T_{\text{compute}}, T_{\text{DMA}}) + T_{\text{final}}
$$

## 11.6 DMA 链表与批处理

### 11.6.1 DMA 链表结构

DMA 链表允许硬件自动执行一系列传输：

```
DMA 链表节点结构：
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Descriptor 1 │────▶│ Descriptor 2 │────▶│ Descriptor 3 │
│ Next_ptr ────┼─────│ Next_ptr ────┼─────│ Next_ptr=NULL│
└──────────────┘     └──────────────┘     └──────────────┘
```

链表节点定义：
$$
\text{Node} = \{\text{desc}, \text{next\_ptr}, \text{flags}, \text{interrupt\_enable}\}
$$

### 11.6.2 批处理优化策略

**命令合并**：将多个小传输合并为批次：

$$
\text{Batch\_efficiency} = \frac{\sum_{i} \text{transfer\_size}_i}{\text{num\_commands} \cdot \text{command\_overhead} + \sum_{i} \text{transfer\_size}_i}
$$

**依赖链管理**：

$$
\text{DAG}_{\text{DMA}} = (V, E)
$$
其中 $V$ 是 DMA 操作集合，$E$ 是依赖边。

拓扑排序生成执行序列：
$$
\text{Schedule} = \text{TopSort}(\text{DAG}_{\text{DMA}})
$$

### 11.6.3 优先级调度

多队列 DMA 的优先级调度：

$$
\text{Priority}(t) = w_1 \cdot \text{urgency}(t) + w_2 \cdot \text{size}(t) + w_3 \cdot \text{locality}(t)
$$

其中：
- $\text{urgency}(t)$：任务紧急度
- $\text{size}(t)$：传输大小的倒数（优先小传输）
- $\text{locality}(t)$：数据局部性得分

### 11.6.4 错误处理与恢复

DMA 错误检测与恢复机制：

**错误类型分类**：
1. 地址越界：$\text{addr} \notin [\text{base}, \text{base} + \text{size})$
2. 对齐错误：$\text{addr} \bmod \text{align\_requirement} \neq 0$
3. 权限错误：访问受保护内存区域

**重试策略**：

$$
\text{Retry\_delay}(n) = \min(2^n \cdot \text{base\_delay}, \text{max\_delay})
$$

其中 $n$ 是重试次数。

**检查点机制**：

$$
\text{Checkpoint} = \{\text{completed\_transfers}, \text{partial\_state}, \text{timestamp}\}
$$

## 本章小结

本章系统介绍了 AI 编译器中多维 Stride DMA 的利用技术。主要内容包括：

1. **多维 DMA 描述符设计**：介绍了从传统一维 DMA 到多维 DMA 的演进，以及描述符的优化策略
2. **Stride 访问模式分析**：分析了不同 stride 模式对性能的影响，以及优化变换方法
3. **张量切片与 DMA 映射**：探讨了如何将高层张量操作高效映射到 DMA 指令
4. **非连续内存传输优化**：包括 gather/scatter、稀疏张量传输和内存访问模式优化
5. **2D/3D DMA 编程模型**：详细介绍了多维 DMA 的编程抽象和实际应用
6. **DMA 链表与批处理**：讨论了提高 DMA 效率的高级技术

关键公式回顾：
- 多维地址计算：$\text{addr}(i_0, ..., i_{n-1}) = \text{base} + \sum_{d=0}^{n-1} i_d \cdot \text{stride}[d]$
- 带宽利用率：$\eta = \frac{\text{useful\_data}}{\text{total\_transactions} \times \text{bus\_width}}$
- 双缓冲收益：$T_{\text{saved}} = \min(T_{\text{compute}}, T_{\text{DMA}}) \times (n-1)$

## 练习题

### 基础题

**练习 11.1**：给定一个 4×6 的矩阵存储在行主序内存中，元素大小为 4 字节。设计一个 2D DMA 描述符来提取 2×3 的子矩阵，起始位置为 (1,2)。

*Hint*：计算源地址偏移和 stride 值。

<details>
<summary>参考答案</summary>

源基地址偏移 = (1×6 + 2) × 4 = 32 字节
DMA 描述符：
- size[0] = 3（列数）
- size[1] = 2（行数）
- src_stride[0] = 4 字节（元素间距）
- src_stride[1] = 24 字节（行间距 = 6×4）
- dst_stride[0] = 4 字节
- dst_stride[1] = 12 字节（目标行间距 = 3×4）

</details>

**练习 11.2**：计算 stride 为 17 的访问模式在 16 个 bank 的内存系统中的冲突概率。

*Hint*：使用 gcd 公式计算冲突概率。

<details>
<summary>参考答案</summary>

$P_{\text{conflict}} = \frac{1}{\gcd(17, 16)} = \frac{1}{1} = 1$

由于 gcd(17, 16) = 1，所以没有 bank 冲突，冲突概率为 0。这是因为 17 和 16 互质，访问会均匀分布在所有 bank 上。

</details>

**练习 11.3**：对于 im2col 操作，输入特征图大小为 32×32，卷积核大小为 3×3，stride=2。计算输出矩阵的维度和所需的 DMA 传输次数。

*Hint*：计算滑窗数量和每个滑窗的元素数。

<details>
<summary>参考答案</summary>

输出特征图大小 = $\lceil \frac{32-3}{2} \rceil + 1 = 15 + 1 = 16$
滑窗总数 = 16 × 16 = 256
每个滑窗元素数 = 3 × 3 = 9
输出矩阵维度 = 9 × 256

如果使用 2D DMA，需要 256 次传输（每个滑窗一次）。
如果优化为批处理，可以减少到更少的 DMA 操作。

</details>

### 挑战题

**练习 11.4**：设计一个 DMA 调度算法，实现矩阵转置操作的最优内存访问模式。矩阵大小为 M×N，cache line 大小为 64 字节，元素大小为 4 字节。

*Hint*：考虑分块策略和 cache line 对齐。

<details>
<summary>参考答案</summary>

最优策略是使用分块转置，块大小选择为 16×16（cache line 可容纳 16 个 float）：

1. 将矩阵划分为 $\lceil \frac{M}{16} \rceil \times \lceil \frac{N}{16} \rceil$ 个块
2. 对每个块内部进行转置（利用 cache 局部性）
3. DMA 描述符设置：
   - 2D DMA，每次传输 16×16 块
   - src_stride[0] = 4, src_stride[1] = N×4
   - dst_stride[0] = M×4, dst_stride[1] = 4
4. 使用双缓冲隐藏传输延迟

这种方法最小化 cache miss，每个 cache line 的数据都被充分利用。

</details>

**练习 11.5**：给定一个稀疏矩阵，非零元素占比 5%，设计一个 DMA 传输策略，比较 CSR 格式和 COO 格式的传输效率。

*Hint*：分析不同格式的内存占用和传输次数。

<details>
<summary>参考答案</summary>

设矩阵大小 M×N，非零元素数 nnz = 0.05×M×N

CSR 格式：
- values 数组：nnz × sizeof(float) = 0.05MN × 4 字节
- col_idx 数组：nnz × sizeof(int) = 0.05MN × 4 字节
- row_ptr 数组：(M+1) × sizeof(int) = (M+1) × 4 字节
- 总传输量：0.1MN × 4 + (M+1) × 4 字节
- DMA 次数：3次（三个数组）

COO 格式：
- values 数组：nnz × sizeof(float) = 0.05MN × 4 字节
- row_idx 数组：nnz × sizeof(int) = 0.05MN × 4 字节
- col_idx 数组：nnz × sizeof(int) = 0.05MN × 4 字节
- 总传输量：0.15MN × 4 字节
- DMA 次数：3次（三个数组）

结论：CSR 格式在 M << N 时更高效，传输量少约 33%。但 COO 格式更灵活，适合动态稀疏模式。

</details>

**练习 11.6**：分析双缓冲 DMA 在不同计算/传输时间比下的性能提升。设计一个自适应算法，根据运行时测量动态选择缓冲区数量。

*Hint*：建立性能模型，考虑缓冲区开销。

<details>
<summary>参考答案</summary>

设 $r = \frac{T_{\text{compute}}}{T_{\text{DMA}}}$

性能提升率：
$$\text{Speedup} = \frac{T_{\text{sequential}}}{T_{\text{pipelined}}} = \frac{n(T_c + T_d)}{T_c + nT_d} \quad (r < 1)$$

$$\text{Speedup} = \frac{n(T_c + T_d)}{nT_c + T_d} \quad (r > 1)$$

自适应算法：
1. 初始使用双缓冲
2. 测量前 k 次迭代的 $T_c$ 和 $T_d$
3. 计算 $r = T_c / T_d$
4. 根据 r 值选择缓冲区数：
   - r < 0.5：单缓冲（传输占主导）
   - 0.5 ≤ r ≤ 2：双缓冲（平衡）
   - r > 2：三缓冲（计算占主导）
5. 考虑内存约束：$n_{\text{buffers}} \times \text{buffer\_size} \leq \text{available\_memory}$

</details>

**练习 11.7**（开放题）：设计一个编译器优化 pass，自动识别代码中的内存访问模式并生成优化的 DMA 指令。考虑如何处理动态 shape 和条件分支。

*Hint*：考虑静态分析、profile-guided optimization 和运行时特化。

<details>
<summary>参考答案</summary>

优化 pass 设计：

1. **静态分析阶段**：
   - 识别循环嵌套中的数组访问模式
   - 提取 stride 信息和访问范围
   - 构建访问模式的符号表示

2. **模式匹配**：
   - 连续访问 → 1D DMA
   - 规则 stride → 2D DMA
   - 嵌套循环 → 3D DMA
   - 间接索引 → gather/scatter DMA

3. **动态 shape 处理**：
   - 生成参数化 DMA 描述符模板
   - 运行时根据实际 shape 实例化
   - 使用符号执行推导 stride 表达式

4. **条件分支处理**：
   - 预测分支概率（profile-guided）
   - 为高概率路径生成 DMA
   - 低概率路径使用标准加载/存储

5. **代价模型**：
   - 估算 DMA 设置开销 vs 收益
   - 只对超过阈值大小的传输使用 DMA
   - 考虑对齐和 cache 影响

6. **验证与回退**：
   - 生成运行时检查确保正确性
   - 提供标准内存访问的回退路径

</details>

## 常见陷阱与错误

1. **Stride 计算错误**
   - 错误：混淆元素步长和字节步长
   - 正确：始终明确单位，建议统一使用字节

2. **边界处理不当**
   - 错误：假设数据总是对齐的
   - 正确：添加边界检查和 padding 处理

3. **忽略 Bank 冲突**
   - 错误：使用 2 的幂次 stride
   - 正确：选择与 bank 数互质的 stride

4. **DMA 链表循环引用**
   - 错误：链表节点错误指向形成环
   - 正确：使用静态分析工具检测循环

5. **缓冲区大小估算错误**
   - 错误：只考虑单个传输的大小
   - 正确：考虑流水线深度和并发传输

6. **同步机制缺失**
   - 错误：假设 DMA 立即完成
   - 正确：使用 fence 和中断确保同步

## 最佳实践检查清单

### 设计阶段
- [ ] 分析目标硬件的 DMA 能力（维度、对齐要求、最大传输大小）
- [ ] 识别应用中的主要数据传输模式
- [ ] 评估 DMA 设置开销 vs 传输收益
- [ ] 设计合适的数据布局以优化 DMA 效率

### 实现阶段
- [ ] 使用描述符池避免频繁分配
- [ ] 实现双缓冲或多缓冲机制
- [ ] 添加地址对齐和边界检查
- [ ] 处理 DMA 错误和异常情况

### 优化阶段
- [ ] 合并小传输减少开销
- [ ] 调整 stride 避免 bank 冲突
- [ ] 使用 DMA 链表批处理操作
- [ ] Profile 实际传输效率并调优

### 验证阶段
- [ ] 验证所有边界条件
- [ ] 测试不同数据大小和 stride 模式
- [ ] 检查内存一致性和同步
- [ ] 性能回归测试确保优化有效