# 第 3 章：计算图表示与分析

计算图是 AI 编译器的核心数据结构，它将神经网络模型转化为可分析、可优化的中间表示。本章深入探讨计算图的构建方法、依赖关系分析、生命周期管理以及别名分析技术。这些技术是后续优化（如算子融合、内存规划、并行化）的基础。对于自动驾驶和具身智能场景，我们特别关注动态控制流、多模态数据流以及超大规模模型的图表示挑战。

## 3.1 数据流图构建

### 3.1.1 节点与边的定义

在 AI 编译器中，计算图 $\mathcal{G} = (V, E)$ 由节点集 $V$ 和边集 $E$ 构成。每个节点 $v \in V$ 表示一个算子（operator），每条边 $e \in E$ 表示数据依赖关系。

**节点属性**：
- 算子类型：$op\_type(v) \in \{Conv, MatMul, Add, ReLU, ...\}$
- 输入张量形状：$input\_shapes(v) = \{s_1, s_2, ..., s_n\}$
- 输出张量形状：$output\_shapes(v) = \{s'_1, s'_2, ..., s'_m\}$  
- 设备放置：$device(v) \in \{CPU, GPU_0, GPU_1, ..., NPU\}$
- 精度要求：$dtype(v) \in \{fp32, fp16, int8, ...\}$

**边属性**：
- 张量元数据：$tensor(e) = (shape, dtype, layout)$
- 数据量：$size(e) = \prod_{i} shape_i \times sizeof(dtype)$
- 传输开销：$cost(e) = f(size(e), bandwidth(src, dst))$

```
     [Input: Image]
           |
           v
      [Conv2D_1]
           |
           v
       [ReLU_1]
           |
           v
      [Conv2D_2]
          / \
         /   \
        v     v
   [MaxPool] [AvgPool]
        \     /
         \   /
          v v
       [Concat]
           |
           v
       [Output]
```

### 3.1.2 静态单赋值（SSA）在计算图中的应用

SSA 形式确保每个变量只被赋值一次，这在计算图中天然成立——每个节点产生新的输出张量。SSA 带来的优势：

1. **简化依赖分析**：每个张量有唯一的定义点
2. **便于优化**：常量传播、死代码消除变得直观
3. **并行化友好**：无需考虑写后写（WAW）冲突

对于需要原地更新的操作（如 BatchNorm 的 running_mean 更新），我们引入 $\phi$ 节点：

$$\phi(t_{old}, t_{new}) = \begin{cases}
t_{old} & \text{if not training} \\
t_{new} & \text{if training}
\end{cases}$$

### 3.1.3 常见算子的图表示

**线性算子**（矩阵乘法）：
$$Y = XW + b$$

图表示：
```
    [X]   [W]   [b]
      \    |    /
       \   |   /
        [MatMul]
           |
         [Add]
           |
          [Y]
```

**卷积算子**：
$$Y_{n,c,h,w} = \sum_{k,r,s} X_{n,k,h+r,w+s} \cdot W_{c,k,r,s} + b_c$$

其中考虑 padding、stride、dilation 等参数。

**注意力机制**（Transformer 核心）：
$$Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

图表示需要展开为多个基础算子：
```
    [Q]    [K]    [V]
     |      |      |
     |   [Trans]   |
     |      |      |
      \    /       |
      [MatMul]     |
         |         |
      [Scale]      |
         |         |
     [Softmax]     |
         \        /
          \      /
          [MatMul]
             |
         [Output]
```

### 3.1.4 自动驾驶场景的多模态输入建模

自动驾驶系统通常融合多种传感器数据：

```
   [Camera]  [LiDAR]  [Radar]  [IMU]
       |        |        |       |
   [CNN_Enc] [PC_Enc] [FFT]  [LSTM]
       |        |        |       |
       +--------+--------+-------+
                |
          [Fusion_Net]
                |
         [Detection_Head]
              /   \
             /     \
    [Trajectory]  [Control]
```

**时序对齐挑战**：不同传感器的采样率不同
- Camera: 30 Hz
- LiDAR: 10 Hz  
- Radar: 20 Hz
- IMU: 100 Hz

需要在图中插入缓冲和插值节点：

$$X_{aligned}(t) = \sum_{i} w_i(t) \cdot X_{sensor_i}(t_i)$$

其中 $w_i(t)$ 是时间权重函数。

## 3.2 控制流与数据依赖

### 3.2.1 条件分支的表示

AI 模型中的条件执行（如 Gated 机制、动态路由）需要特殊的图结构。我们采用 If-Then-Else 子图：

```
        [Condition]
           /    \
          /      \
    [True_Branch] [False_Branch]
          \      /
           \    /
          [Merge]
```

**谓词执行优化**：对于简单条件，可以转换为无分支形式：
$$Y = condition \cdot Y_{true} + (1 - condition) \cdot Y_{false}$$

这避免了控制流开销，但增加了计算量。需要根据分支概率 $p_{true}$ 和分支计算成本 $C_{true}, C_{false}$ 权衡：

$$Cost_{branch} = p_{true} \cdot C_{true} + (1-p_{true}) \cdot C_{false} + C_{control}$$
$$Cost_{predicate} = C_{true} + C_{false} + C_{merge}$$

### 3.2.2 循环结构的处理

循环在 RNN、迭代式算法中普遍存在。循环展开（unrolling）是常见优化：

**完全展开**（适用于固定迭代次数）：
```
Original:
    [Init]
      |
    [Loop_Body] <--+
      |           |
    [Update]------+
      |
    [Output]

Unrolled:
    [Init]
      |
    [Body_1]
      |
    [Body_2]
      |
     ...
      |
    [Body_N]
      |
    [Output]
```

**部分展开**（平衡代码膨胀与性能）：
展开因子 $k$ 的选择依据：
- 寄存器压力：$R_{available} \geq k \cdot R_{per\_iteration}$
- 缓存容量：$L1_{size} \geq k \cdot Code_{size}$
- 指令级并行度：$ILP_{potential} = min(k, IssueWidth)$

### 3.2.3 依赖关系分类

**真依赖（RAW - Read After Write）**：
$$v_j \text{ uses output of } v_i \Rightarrow (v_i, v_j) \in E_{true}$$

**反依赖（WAR - Write After Read）**：
在原地操作中需要考虑：
$$v_j \text{ overwrites input of } v_i \Rightarrow \text{需要额外约束}$$

**输出依赖（WAW - Write After Write）**：
SSA 形式自动消除，但在内存复用时需要考虑。

**依赖距离分析**：
对于循环中的依赖，定义依赖距离向量：
$$\vec{d} = (d_1, d_2, ..., d_n)$$
其中 $d_i$ 表示第 $i$ 维循环的迭代距离。

### 3.2.4 具身智能中的动态控制流

具身智能系统的感知-决策-控制循环具有高度动态性：

```
    [Perception]
         |
    [World_Model]
         |
    [Policy_Net]
       /   \
      /     \
[Explore] [Exploit]  <-- 动态选择
      \     /
       \   /
    [Action_Gen]
         |
    [Safety_Check]
       /   \
      /     \
[Execute] [Fallback]
```

**编译挑战**：
1. 分支预测困难：行为依赖于环境状态
2. 延迟约束严格：控制频率通常 > 100 Hz
3. 安全性要求：需要保证 worst-case 执行时间

**解决方案**：
- 投机编译多个可能路径
- 运行时特化（JIT）
- 保守的内存预分配

## 3.3 活性分析与生命周期

### 3.3.1 定义-使用链

定义-使用链（def-use chain）连接张量的产生点和所有使用点：

$$DU(t) = \{v \in V | v \text{ uses tensor } t\}$$

使用-定义链（use-def chain）反向追溯：

$$UD(v, i) = \text{节点 } v \text{ 第 } i \text{ 个输入的来源}$$

**链的构建算法**：
1. 遍历计算图，记录每个张量的定义节点
2. 对每个节点的输入，建立到定义节点的链接
3. 维护引用计数用于后续内存管理

### 3.3.2 活性区间计算

张量 $t$ 的活性区间 $[birth(t), death(t)]$ 定义为：
- $birth(t)$：产生 $t$ 的节点的拓扑序号
- $death(t)$：最后使用 $t$ 的节点的拓扑序号

**扩展活性分析**（考虑并行执行）：
$$LiveInterval(t) = [earliest\_start(def(t)), latest\_end(uses(t))]$$

其中考虑了节点的并行调度可能性。

**活性密度图**：
```
Tensor  |-------- Lifetime --------|
  A     |████████████░░░░░░░░░░░░░░|
  B     |░░░░████████████░░░░░░░░░░|
  C     |░░░░░░░░████████████░░░░░░|
  D     |░░████████░░░░████████░░░░|
        +---------------------------+
        0                          Time
```

### 3.3.3 内存压力分析

在时刻 $t$ 的内存压力：
$$M(t) = \sum_{tensor \in Live(t)} size(tensor)$$

**峰值内存**：
$$M_{peak} = \max_t M(t)$$

**内存压力缓解策略**：
1. **重计算**：丢弃中间结果，需要时重新计算
   - 收益：$\Delta M = size(tensor)$
   - 代价：$\Delta T = compute\_time(tensor)$
   
2. **异步传输**：利用 DMA 重叠计算与传输
   - 条件：$transfer\_time < compute\_time_{next}$

3. **算子融合**：减少中间张量
   - 示例：$Conv \rightarrow BN \rightarrow ReLU$ 融合可省去两个中间张量

### 3.3.4 200T 模型的生命周期优化

超大规模模型的特殊挑战：

**分层生命周期管理**：
- L0：寄存器（~KB）- 单算子内
- L1：片上缓存（~MB）- 算子间
- L2：HBM（~GB）- 层间
- L3：主存（~TB）- 模型分片间
- L4：SSD（~PB）- 检查点

**生命周期策略**：
```
if size(tensor) < L1_threshold:
    keep_in_cache()
elif size(tensor) < L2_threshold and reuse_distance < threshold:
    keep_in_HBM()
elif is_checkpoint(tensor):
    offload_to_SSD()
else:
    recompute_when_needed()
```

**混合精度生命周期**：
不同精度张量的生命周期管理：
- FP32 权重：长生命周期，常驻内存
- FP16 激活：中等生命周期，可重计算
- INT8 中间结果：短生命周期，积极回收

## 3.4 别名分析基础

### 3.4.1 张量视图与切片

张量视图（view）共享底层存储但具有不同的逻辑形状或步长：

$$View(T, shape', stride') = \{T_{base}, offset, shape', stride'\}$$

**视图操作分类**：
1. **Reshape**：改变形状但保持元素顺序
   - 条件：$\prod shape = \prod shape'$
   - 连续性要求：原张量必须连续

2. **Transpose**：改变轴顺序
   - 新步长：$stride'[i] = stride[perm[i]]$

3. **Slice**：提取子张量
   - 新偏移：$offset' = offset + \sum_i start_i \times stride_i$

4. **Broadcast**：扩展维度
   - 零步长维度：$stride'[i] = 0$ for broadcasted dims

### 3.4.2 Stride 张量的别名判定

两个张量 $T_1$ 和 $T_2$ 存在别名当且仅当它们的内存区间有重叠：

$$Alias(T_1, T_2) \Leftrightarrow [base_1, base_1 + size_1) \cap [base_2, base_2 + size_2) \neq \emptyset$$

**精确别名分析**（适用于规则步长）：

对于多维张量，需要分析索引空间的重叠：
$$\exists (i_1, ..., i_n), (j_1, ..., j_m): addr(T_1, i_1, ..., i_n) = addr(T_2, j_1, ..., j_m)$$

其中地址计算：
$$addr(T, i_1, ..., i_n) = base + \sum_{k=1}^n i_k \times stride_k$$

**区间分析方法**：
1. 计算每个张量的地址范围
2. 检查范围重叠
3. 对于重叠区域，验证是否存在有效索引

### 3.4.3 指向分析在 AI 编译器中的应用

AI 编译器的指向分析相对简化，因为：
- 无指针算术
- 张量生命周期明确
- 无递归数据结构

**别名类别**：
1. **Must-Alias**：确定别名（如显式 view）
2. **May-Alias**：可能别名（如动态索引）
3. **No-Alias**：确定无别名（不同基址）

**别名图构建**：
```python
AliasGraph = {
    tensor_id: {
        'base': base_tensor,
        'may_alias': set(),
        'must_alias': set(),
        'no_alias': set()
    }
}
```

### 3.4.4 原地操作的安全性检查

原地操作 $T = op(T, ...)$ 需要满足：

1. **无其他引用**：$refcount(T) = 1$
2. **无活跃视图**：$\forall V \in Views(T): dead(V)$
3. **依赖满足**：所有读操作在写之前完成

**安全性验证算法**：
```
function is_safe_inplace(op, tensor):
    if refcount(tensor) > 1:
        return False
    for view in get_views(tensor):
        if is_live(view):
            return False
    for user in get_users(tensor):
        if not is_scheduled_before(user, op):
            return False
    return True
```

**自动驾驶场景的特殊考虑**：
- 传感器数据缓冲区的循环使用
- 历史帧的引用管理
- 安全关键路径的数据隔离

## 本章小结

本章系统介绍了 AI 编译器中计算图的表示与分析技术：

**核心概念**：
1. 计算图 $\mathcal{G} = (V, E)$ 是模型的中间表示
2. SSA 形式简化了依赖分析和优化
3. 控制流需要特殊的图结构处理
4. 活性分析决定了内存分配策略
5. 别名分析确保了优化的正确性

**关键公式**：
- 内存压力：$M(t) = \sum_{tensor \in Live(t)} size(tensor)$
- 依赖距离：$\vec{d} = (d_1, d_2, ..., d_n)$
- 地址计算：$addr(T, i_1, ..., i_n) = base + \sum_{k=1}^n i_k \times stride_k$
- 别名条件：$[base_1, base_1 + size_1) \cap [base_2, base_2 + size_2) \neq \emptyset$

**实践要点**：
- 多模态融合需要考虑时序对齐
- 动态控制流需要运行时特化
- 超大模型需要分层生命周期管理
- 原地操作需要严格的安全性检查

这些分析技术为后续的内存优化、算子融合、并行化等高级优化奠定了基础。

## 练习题

### 基础题

**练习 3.1**：给定如下计算序列，构建对应的数据流图并标注张量形状。
```
X: [32, 3, 224, 224]  # 批量大小32，3通道，224x224图像
W1: [64, 3, 7, 7]     # 64个7x7卷积核
Y1 = Conv2D(X, W1, stride=2, padding=3)
Y2 = BatchNorm(Y1)
Y3 = ReLU(Y2)
Y4 = MaxPool2D(Y3, kernel=3, stride=2)
```

*Hint*: 注意计算每步输出的形状变化，特别是 stride 和 padding 的影响。

<details>
<summary>答案</summary>

数据流图：
```
[X] → [Conv2D] → [Y1] → [BatchNorm] → [Y2] → [ReLU] → [Y3] → [MaxPool2D] → [Y4]
        ↑
      [W1]
```

形状推导：
- Y1: Conv2D 输出 = $\lfloor \frac{224 + 2 \times 3 - 7}{2} \rfloor + 1 = 112$，形状 [32, 64, 112, 112]
- Y2: BatchNorm 不改变形状，[32, 64, 112, 112]
- Y3: ReLU 不改变形状，[32, 64, 112, 112]
- Y4: MaxPool 输出 = $\lfloor \frac{112 - 3}{2} \rfloor + 1 = 55$，形状 [32, 64, 55, 55]

</details>

**练习 3.2**：计算下列张量操作序列的峰值内存占用（假设 FP32）。
```
A = torch.randn(1024, 1024)  # 4MB
B = torch.randn(1024, 1024)  # 4MB
C = A @ B                     # 4MB
D = C + A                     # 4MB
E = D.sum(dim=0)             # 4KB
del A, B                     # 释放A、B
F = D @ C                     # 4MB
```

*Hint*: 画出每个张量的生命周期，找出同时存活的最大集合。

<details>
<summary>答案</summary>

生命周期分析：
```
时刻  操作        存活张量        内存占用
1    创建A       {A}             4MB
2    创建B       {A,B}           8MB
3    C=A@B       {A,B,C}         12MB
4    D=C+A       {A,B,C,D}       16MB  ← 峰值
5    E=D.sum     {A,B,C,D,E}     16MB+4KB ≈ 16MB
6    del A,B     {C,D,E}         8MB+4KB
7    F=D@C       {C,D,E,F}       12MB+4KB
```

峰值内存：16MB（时刻4）

</details>

**练习 3.3**：判断以下张量对是否可能存在别名关系。
```python
base = torch.randn(100, 200)  # 基础张量
a = base[10:20, :]            # 切片
b = base[15:25, :]            # 切片
c = base.T                    # 转置
d = base.reshape(200, 100)    # 重塑
e = torch.randn(100, 200)     # 新张量
```

*Hint*: 切片操作共享底层存储，reshape 需要连续性。

<details>
<summary>答案</summary>

别名关系分析：
- (a, b): **Must-Alias** - 两个切片区间 [10:20] 和 [15:25] 有重叠 [15:20]
- (a, c): **Must-Alias** - 都引用 base 的数据
- (a, d): **Must-Alias** - reshape 返回视图（base 连续）
- (b, c): **Must-Alias** - 都引用 base 的数据
- (b, d): **Must-Alias** - 都引用 base 的数据
- (c, d): **Must-Alias** - 都引用 base 的数据
- (e, 其他): **No-Alias** - e 是独立分配的新张量

</details>

### 挑战题

**练习 3.4**：设计一个算法，检测计算图中的循环依赖。给定邻接表表示的有向图，返回是否存在循环以及循环路径。

*Hint*: 使用 DFS 配合三色标记（白、灰、黑）。

<details>
<summary>答案</summary>

算法设计：
1. 白色：未访问
2. 灰色：正在访问（在当前 DFS 路径上）
3. 黑色：已完成访问

检测算法：
```
function detect_cycle(graph):
    color = {v: WHITE for v in graph}
    parent = {v: None for v in graph}
    
    function dfs(v):
        color[v] = GRAY
        for u in graph[v]:
            if color[u] == GRAY:
                # 找到循环，回溯路径
                cycle = [u, v]
                p = parent[v]
                while p != u:
                    cycle.append(p)
                    p = parent[p]
                return cycle
            elif color[u] == WHITE:
                parent[u] = v
                result = dfs(u)
                if result:
                    return result
        color[v] = BLACK
        return None
    
    for v in graph:
        if color[v] == WHITE:
            cycle = dfs(v)
            if cycle:
                return True, cycle
    return False, None
```

时间复杂度：O(V + E)

</details>

**练习 3.5**：推导 Transformer 注意力机制的内存占用公式，考虑序列长度 $n$、批量大小 $b$、隐藏维度 $d$。分析 FlashAttention 如何减少内存占用。

*Hint*: 标准注意力需要存储 $QK^T$ 矩阵，FlashAttention 使用分块计算。

<details>
<summary>答案</summary>

标准注意力内存分析：

输入：
- Q, K, V: 各 $b \times n \times d$

中间结果：
- $QK^T$: $b \times n \times n$
- Softmax 输出: $b \times n \times n$
- 最终输出: $b \times n \times d$

总内存（FP32）：
$$M_{standard} = 4 \times (3bnd + 2bn^2 + bnd) = 4b(4nd + 2n^2)$$

当 $n >> d$ 时，$2bn^2$ 项主导，内存复杂度 $O(bn^2)$。

FlashAttention 优化：
- 将 Q, K, V 分块：块大小 $B_r \times B_c$
- 每次只计算一个块的注意力
- 使用在线 softmax 避免存储完整矩阵

分块内存：
$$M_{flash} = 4b(4nd + 2B_r B_c)$$

内存减少比例：
$$\frac{M_{standard}}{M_{flash}} \approx \frac{n^2}{B_r B_c}$$

典型配置 $B_r = B_c = 64$，对于 $n = 2048$，内存减少 $\frac{2048^2}{64^2} = 1024$ 倍。

</details>

**练习 3.6**：分析在多 GPU 训练中，如何确定张量的最优放置策略。给定计算图和通信成本模型，形式化为优化问题。

*Hint*: 考虑计算负载均衡和通信最小化的权衡。

<details>
<summary>答案</summary>

优化问题形式化：

决策变量：
- $x_{v,d} \in \{0,1\}$：节点 $v$ 是否放置在设备 $d$

目标函数：
$$\min \alpha T_{comp} + \beta T_{comm}$$

其中：
- 计算时间：$T_{comp} = \max_d \sum_v x_{v,d} \cdot t_v$
- 通信时间：$T_{comm} = \sum_{(u,v) \in E} \sum_{d_1 \neq d_2} x_{u,d_1} \cdot x_{v,d_2} \cdot c_{u,v}$

约束条件：
1. 每个节点恰好放置在一个设备：$\sum_d x_{v,d} = 1, \forall v$
2. 内存约束：$\sum_v x_{v,d} \cdot m_v \leq M_d, \forall d$
3. 依赖约束：关键路径节点优先级

这是一个整数线性规划（ILP）问题，NP-hard。

实践解法：
1. 贪心算法：基于计算/通信比率
2. 动态规划：对于链式结构
3. 启发式搜索：模拟退火、遗传算法
4. 强化学习：学习放置策略

</details>

**练习 3.7**：设计一个内存分配算法，给定张量生命周期，最小化内存碎片。考虑不同大小的张量和对齐要求。

*Hint*: 可以借鉴操作系统的内存管理算法，如 Best-Fit、Buddy System。

<details>
<summary>答案</summary>

算法设计：混合策略内存分配器

数据结构：
```python
class MemoryAllocator:
    def __init__(self, total_size, alignment=64):
        self.free_lists = defaultdict(list)  # size_class -> [blocks]
        self.allocated = {}  # tensor_id -> (offset, size)
        self.alignment = alignment
```

分配策略：
1. **大小分类**：将张量分为小（<1MB）、中（1-16MB）、大（>16MB）

2. **小张量**：使用固定大小池
   - 预定义大小：64KB, 128KB, 256KB, 512KB
   - 减少碎片，快速分配

3. **中张量**：Best-Fit with Coalescing
   ```python
   def allocate_medium(size):
       aligned_size = align_up(size, alignment)
       best_block = find_best_fit(free_lists, aligned_size)
       if best_block:
           split_if_needed(best_block, aligned_size)
       else:
           compact_memory()  # 碎片整理
           best_block = find_best_fit(free_lists, aligned_size)
       return best_block
   ```

4. **大张量**：直接映射
   - 使用 mmap 风格的大页分配
   - 避免碎片化主内存池

5. **碎片整理**：
   - 延迟整理：碎片率 > 25% 时触发
   - 增量整理：每次移动有限数量块

性能指标：
- 外部碎片率：$\frac{总空闲内存 - 最大可分配块}{总空闲内存}$
- 内部碎片率：$\frac{分配内存 - 请求内存}{分配内存}$
- 分配延迟：O(log n) with balanced trees

</details>

**练习 3.8**：给定一个包含动态控制流的模型（如 Neural Architecture Search），设计编译时的形状推导算法，处理形状的符号表达式。

*Hint*: 使用符号数学库，建立约束求解系统。

<details>
<summary>答案</summary>

符号形状推导系统：

1. **符号表示**：
```python
class SymbolicDim:
    def __init__(self, name, constraints=None):
        self.name = name  # e.g., "batch_size", "seq_len"
        self.min = constraints.get('min', 1)
        self.max = constraints.get('max', float('inf'))
        self.divisible_by = constraints.get('divisible_by', 1)
```

2. **形状代数**：
```python
# 卷积输出形状
def conv_shape(input_shape, kernel, stride, padding):
    H_in = input_shape[2]
    H_out = SymbolicExpr(
        (H_in + 2*padding - kernel) // stride + 1
    )
    return [input_shape[0], num_filters, H_out, W_out]
```

3. **约束传播**：
```python
class ConstraintSystem:
    def add_equality(self, expr1, expr2):
        # MatMul: (M, K) @ (K, N) -> (M, N)
        # 约束: expr1.shape[1] == expr2.shape[0]
        
    def add_inequality(self, expr, bound):
        # 内存约束: prod(shape) * dtype_size <= memory_limit
        
    def solve(self):
        # 使用 Z3 或类似求解器
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        if solver.check() == z3.sat:
            return solver.model()
```

4. **动态分支处理**：
```python
def analyze_conditional(cond, true_branch, false_branch):
    # 收集两个分支的形状约束
    true_shapes = analyze_branch(true_branch)
    false_shapes = analyze_branch(false_branch)
    
    # 合并约束（输出形状必须兼容）
    output_shape = unify_shapes(true_shapes, false_shapes)
    
    # 生成运行时检查
    runtime_checks = generate_shape_checks(output_shape)
    
    return output_shape, runtime_checks
```

5. **实例：NAS 中的动态深度**
```python
depth = SymbolicDim("depth", {"min": 1, "max": 4})
for i in range(depth):
    if search_space[i]:  # 动态选择
        x = conv_block(x)
    else:
        x = identity(x)

# 推导：输出形状依赖于 depth 和 search_space
# 编译策略：
# 1. 为每个可能的深度生成代码
# 2. 运行时根据实际深度选择
```

关键技术：
- 区间算术：处理形状范围
- 模运算：处理对齐约束
- 不动点迭代：处理循环结构
- 概率推理：基于 profile 预测常见形状

</details>

## 常见陷阱与错误

### 1. 图构建陷阱

**陷阱**：忽略隐式依赖
```
# 错误：未考虑 BatchNorm 的 running_mean 更新
bn_output = BatchNorm(input)  # 还有 running_mean 的副作用
```
**解决**：显式建模所有状态更新为图中的边。

### 2. 内存分析错误

**陷阱**：低估峰值内存
```
# 错误：只考虑输入输出，忽略中间梯度
peak_memory = input_size + output_size
```
**正确**：考虑反向传播时的梯度累积。

### 3. 别名分析盲点

**陷阱**：Reshape 后的连续性假设
```
# 错误：假设 reshape 总是返回视图
view = tensor.T.reshape(...)  # T 破坏连续性，reshape 会复制！
```
**检查**：使用 `is_contiguous()` 验证。

### 4. 活性分析的并行陷阱

**陷阱**：串行假设下的生命周期
```
# 错误：假设严格的顺序执行
lifetime = [def_point, last_use_point]
```
**正确**：考虑算子的并行执行可能。

### 5. 动态形状的编译时假设

**陷阱**：过度特化
```
# 错误：为每个见过的形状生成代码
if shape == (32, 224, 224):
    use_optimized_kernel_32_224_224()
```
**解决**：使用形状类别和参数化核函数。

## 最佳实践检查清单

### 图构建阶段
- [ ] 所有算子的输入输出形状都已验证
- [ ] 隐式状态更新（如 BN 的 running stats）已建模
- [ ] 控制流依赖已正确表示
- [ ] 数据类型（dtype）和布局（layout）已标注
- [ ] 设备放置（device placement）已确定

### 依赖分析阶段
- [ ] 真依赖（RAW）完整识别
- [ ] 反依赖（WAR）和输出依赖（WAW）已处理
- [ ] 控制依赖已加入调度约束
- [ ] 跨设备依赖的通信开销已量化

### 生命周期管理
- [ ] 张量生命周期的起止点准确
- [ ] 考虑了并行执行对生命周期的影响
- [ ] 峰值内存估算包含所有临时缓冲区
- [ ] 梯度累积的内存开销已计入
- [ ] 设置了合理的重计算策略

### 别名分析
- [ ] 所有视图操作的别名关系已记录
- [ ] 原地操作的安全性已验证
- [ ] 跨函数的别名传播已处理
- [ ] 动态索引的别名可能性已分析

### 性能优化检查
- [ ] 识别了内存带宽瓶颈
- [ ] 标记了可融合的算子序列
- [ ] 评估了重计算 vs 存储的权衡
- [ ] 考虑了数据局部性优化机会

### 正确性保证
- [ ] 数值稳定性分析（特别是混合精度）
- [ ] 确定性要求已满足（如安全关键路径）
- [ ] 边界条件和特殊输入已测试
- [ ] 动态形状的运行时检查已插入

### 可维护性
- [ ] 图的可视化输出可用
- [ ] 关键决策有日志记录
- [ ] 性能统计数据可导出
- [ ] 调试模式可追踪中间结果