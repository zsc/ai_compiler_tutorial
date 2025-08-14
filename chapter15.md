# 第 15 章：NUMA 架构优化（一）

## 章节概述

本章深入探讨 NUMA（Non-Uniform Memory Access，非统一内存访问）架构下的 AI 编译器优化技术。随着 200T 参数级模型的出现，单一计算节点已无法满足计算和内存需求，NUMA 架构成为高性能计算的必然选择。本章将从 NUMA 基础概念出发，详细讨论亲和性设置、本地内存分配策略以及 NUMA 感知的数据放置技术，为读者在多 Socket 系统上优化 AI 工作负载提供理论基础和实践指导。

## 15.1 NUMA 基础概念

### 15.1.1 NUMA 架构演进

传统的 SMP（Symmetric Multi-Processing）系统中，所有处理器通过共享总线访问统一的内存空间，这种架构在处理器数量增加时会遇到严重的总线竞争问题。NUMA 架构通过将内存分布到各个处理器节点，每个节点拥有本地内存，从而解决了这一瓶颈。

```
    传统 SMP 架构                    NUMA 架构
    
    CPU0  CPU1  CPU2  CPU3          Node 0          Node 1
      |     |     |     |           ┌─────────┐    ┌─────────┐
      └──┬──┴──┬──┴──┬──┘           │ CPU0-1  │    │ CPU2-3  │
         │     │     │              │ Memory0 │    │ Memory1 │
    ─────┴─────┴─────┴─────         └────┬────┘    └────┬────┘
         Shared Memory                    └──────┬───────┘
                                              QPI/UPI
```

### 15.1.2 内存访问延迟层次

在 NUMA 系统中，内存访问延迟呈现明显的层次结构：

$$L_{access} = \begin{cases}
L_{local} & \text{if } node(CPU) = node(Memory) \\
L_{remote} & \text{if } node(CPU) \neq node(Memory)
\end{cases}$$

其中，$L_{remote} \approx \alpha \cdot L_{local}$，$\alpha$ 通常在 1.5 到 2.5 之间，具体取决于互连技术（如 Intel QPI/UPI、AMD Infinity Fabric）。

### 15.1.3 NUMA 距离矩阵

系统通过 NUMA 距离矩阵描述节点间的访问代价：

$$D = \begin{bmatrix}
d_{00} & d_{01} & \cdots & d_{0n} \\
d_{10} & d_{11} & \cdots & d_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
d_{n0} & d_{n1} & \cdots & d_{nn}
\end{bmatrix}$$

其中 $d_{ij}$ 表示从节点 $i$ 访问节点 $j$ 内存的相对延迟。对角线元素 $d_{ii} = 10$（标准化本地访问），非对角线元素通常为 20 或 21。

### 15.1.4 NUMA 拓扑发现

编译器需要在编译时或运行时发现 NUMA 拓扑结构。关键信息包括：

- **节点数量**：$N_{nodes}$
- **每节点 CPU 数**：$C_{per\_node}$
- **每节点内存容量**：$M_{per\_node}$
- **节点间带宽矩阵**：$B_{ij}$

这些信息通过系统调用（如 Linux 的 `libnuma`）或硬件抽象层获取。

## 15.2 亲和性设置

### 15.2.1 CPU 亲和性

CPU 亲和性控制线程到特定 CPU 核心的绑定，是 NUMA 优化的基础：

$$\text{Affinity}: T \rightarrow \mathcal{P}(C)$$

其中 $T$ 是线程集合，$C$ 是 CPU 核心集合，$\mathcal{P}(C)$ 是 $C$ 的幂集。

亲和性策略包括：

1. **紧密绑定（Compact）**：将线程绑定到同一 NUMA 节点
   - 优势：最小化内存访问延迟
   - 劣势：可能造成节点内资源竞争

2. **分散绑定（Scatter）**：将线程均匀分布到各节点
   - 优势：均衡使用内存带宽
   - 劣势：增加跨节点通信开销

3. **层次绑定（Hierarchical）**：基于任务依赖关系的智能绑定

### 15.2.2 内存亲和性

内存亲和性策略决定数据的物理放置位置：

$$\text{Placement}: M \rightarrow N$$

其中 $M$ 是内存页集合，$N$ 是 NUMA 节点集合。

常见策略：

1. **首次触碰（First-Touch）**：页面分配到首次访问它的线程所在节点
2. **交错放置（Interleave）**：页面轮流分配到各节点
3. **绑定放置（Bind）**：显式指定页面所属节点

### 15.2.3 亲和性优化模型

给定计算图 $\mathcal{G} = (V, E)$ 和 NUMA 拓扑，亲和性优化问题可形式化为：

$$\min_{f: V \rightarrow N} \sum_{(u,v) \in E} w_{uv} \cdot d_{f(u),f(v)} + \sum_{v \in V} c_v \cdot l_{f(v)}$$

其中：
- $w_{uv}$ 是边 $(u,v)$ 的通信量
- $d_{ij}$ 是节点 $i$ 到 $j$ 的 NUMA 距离
- $c_v$ 是顶点 $v$ 的计算量
- $l_i$ 是节点 $i$ 的负载

这是一个 NP-hard 问题，实践中使用启发式算法求解。

## 15.3 本地内存分配策略

### 15.3.1 静态分配策略

编译时确定的内存分配策略，基于程序分析和性能模型：

1. **数据分区**：将大型张量按 NUMA 节点数分区
   $$T_{large} = T_0 \oplus T_1 \oplus \cdots \oplus T_{n-1}$$
   其中 $T_i$ 分配到节点 $i$

2. **复制策略**：对频繁访问的只读数据进行复制
   $$\text{Replicate}(T_{readonly}) = \{T^{(0)}, T^{(1)}, \ldots, T^{(n-1)}\}$$

3. **迁移策略**：基于访问模式的数据迁移
   $$\text{Migrate}(T, n_{src}, n_{dst}) \text{ if } A_{dst}(T) > \theta \cdot A_{src}(T)$$
   其中 $A_i(T)$ 是节点 $i$ 对张量 $T$ 的访问频率

### 15.3.2 动态分配策略

运行时根据系统状态调整的分配策略：

1. **自适应页面迁移**：
   $$P_{migrate} = \frac{R_{remote}}{R_{remote} + R_{local}} > \tau$$
   当远程访问比例超过阈值 $\tau$ 时触发迁移

2. **内存压力均衡**：
   $$\text{Balance}(M_i) = M_{avg} \pm \delta$$
   保持各节点内存使用量在平均值 $M_{avg}$ 的 $\delta$ 范围内

3. **带宽感知分配**：
   $$node_{alloc} = \arg\min_i \frac{BW_{used}^{(i)}}{BW_{max}^{(i)}}$$
   选择带宽利用率最低的节点进行分配

### 15.3.3 大页面优化

使用大页面（Huge Pages）减少 TLB 失效，在 NUMA 系统中尤为重要：

$$\text{TLB\_Coverage} = \text{Page\_Size} \times \text{TLB\_Entries}$$

标准页面（4KB）vs 大页面（2MB/1GB）：
- 减少页表遍历开销
- 降低 TLB 失效率
- 简化 NUMA 页面管理

大页面分配策略需要考虑：
1. 内存碎片化风险
2. NUMA 节点内存容量限制
3. 页面共享与迁移的粒度权衡

## 15.4 NUMA 感知的数据放置

### 15.4.1 张量分布模型

对于大规模张量，需要设计 NUMA 感知的分布策略：

1. **块分布（Block Distribution）**：
   $$T[i:j] \rightarrow node_k \text{ where } k = \lfloor \frac{i \cdot N_{nodes}}{size(T)} \rfloor$$

2. **循环分布（Cyclic Distribution）**：
   $$T[i] \rightarrow node_{(i \mod N_{nodes})}$$

3. **块循环分布（Block-Cyclic）**：
   结合块分布和循环分布的优点
   $$T[b \cdot B + offset] \rightarrow node_{(b \mod N_{nodes})}$$
   其中 $B$ 是块大小

### 15.4.2 计算-数据协同放置

将计算任务与其访问的数据放置在同一 NUMA 节点：

**局部性度量**：
$$\text{Locality}(t, n) = \frac{\sum_{d \in D_t \cap M_n} size(d)}{\sum_{d \in D_t} size(d)}$$

其中 $D_t$ 是任务 $t$ 访问的数据集，$M_n$ 是节点 $n$ 的本地内存。

**优化目标**：
$$\max \sum_{t \in T} \sum_{n \in N} x_{tn} \cdot \text{Locality}(t, n)$$

约束条件：
- $\sum_n x_{tn} = 1$ （每个任务分配到一个节点）
- $\sum_t x_{tn} \cdot comp(t) \leq capacity(n)$ （节点容量限制）

### 15.4.3 多级缓存优化

NUMA 系统的多级缓存层次需要特殊考虑：

```
    L1 Cache (32KB)
         ↓
    L2 Cache (256KB) 
         ↓
    L3 Cache (30MB, shared within socket)
         ↓
    Local Memory (DDR4/5)
         ↓
    Remote Memory (via QPI/UPI)
```

缓存优化策略：
1. **缓存着色（Cache Coloring）**：避免关键数据的缓存冲突
2. **预取优化（Prefetching）**：考虑 NUMA 延迟的预取时机
3. **伪共享消除**：避免跨 NUMA 节点的缓存行共享

### 15.4.4 Transformer 模型的 NUMA 优化

以 Transformer 模型为例，展示 NUMA 感知的数据放置：

1. **注意力矩阵分块**：
   $$A = QK^T = \begin{bmatrix} Q_0 \\ Q_1 \\ \vdots \\ Q_{n-1} \end{bmatrix} \begin{bmatrix} K_0^T & K_1^T & \cdots & K_{n-1}^T \end{bmatrix}$$
   
   将 $Q_i$ 和 $K_i$ 放置在节点 $i$，减少跨节点通信

2. **FFN 层分区**：
   $$\text{FFN}(x) = \text{GELU}(xW_1)W_2$$
   
   将 $W_1$ 按列分区，$W_2$ 按行分区，实现计算的 NUMA 局部性

3. **All-Reduce 优化**：
   使用 NUMA 感知的树形规约算法，优先进行节点内规约

## 本章小结

本章系统介绍了 NUMA 架构下的 AI 编译器优化基础：

1. **NUMA 基础概念**：理解非统一内存访问的本质，掌握 NUMA 距离矩阵和拓扑发现方法
2. **亲和性设置**：通过 CPU 和内存亲和性控制，优化线程和数据的物理放置
3. **本地内存分配**：设计静态和动态分配策略，利用大页面优化减少开销
4. **数据放置优化**：实现张量分布、计算-数据协同放置和多级缓存优化

关键公式回顾：
- 内存访问延迟：$L_{remote} \approx \alpha \cdot L_{local}$
- 亲和性优化：$\min \sum_{(u,v) \in E} w_{uv} \cdot d_{f(u),f(v)}$
- 局部性度量：$\text{Locality}(t, n) = \frac{\text{local\_data}}{\text{total\_data}}$

下一章将继续深入探讨跨 Socket 通信优化、NUMA 平衡算法以及大规模 Transformer 的 NUMA 优化实践。

## 练习题

### 基础题

**练习 15.1**：NUMA 距离矩阵计算

给定一个 4 节点 NUMA 系统，节点间通过环形拓扑连接，相邻节点间延迟为 20ns，本地访问延迟为 10ns。请构建该系统的 NUMA 距离矩阵。

*提示：考虑最短路径，环形拓扑中对角节点需要经过两跳。*

<details>
<summary>答案</summary>

距离矩阵为：
$$D = \begin{bmatrix}
10 & 20 & 30 & 20 \\
20 & 10 & 20 & 30 \\
30 & 20 & 10 & 20 \\
20 & 30 & 20 & 10
\end{bmatrix}$$

解释：
- 对角线元素（本地访问）：10
- 相邻节点（如 0→1, 1→2, 2→3, 3→0）：20
- 对角节点（如 0→2, 1→3）：30（需要经过两跳）

</details>

**练习 15.2**：内存带宽计算

某 NUMA 系统有 2 个节点，每节点本地内存带宽为 100 GB/s，跨节点带宽为 40 GB/s。若一个应用 70% 的内存访问是本地的，30% 是远程的，计算实际可达到的平均内存带宽。

*提示：使用加权平均计算有效带宽。*

<details>
<summary>答案</summary>

平均带宽 = 0.7 × 100 GB/s + 0.3 × 40 GB/s = 70 + 12 = 82 GB/s

带宽效率 = 82 / 100 = 82%

这说明即使只有 30% 的远程访问，也会导致 18% 的带宽损失。

</details>

**练习 15.3**：页面迁移决策

一个 4KB 页面在过去 1000 次访问中，本地访问 200 次（每次 10ns），远程访问 800 次（每次 20ns）。页面迁移开销为 10μs。计算是否应该迁移该页面。

*提示：比较迁移前后的总开销。*

<details>
<summary>答案</summary>

当前总延迟：200 × 10ns + 800 × 20ns = 2000ns + 16000ns = 18μs

迁移后（假设访问模式不变）：
- 迁移开销：10μs
- 新的访问延迟：800 × 10ns + 200 × 20ns = 8000ns + 4000ns = 12μs
- 总开销：10μs + 12μs = 22μs

结论：不应迁移，因为 22μs > 18μs。

但如果考虑未来多次访问，设访问次数为 n：
- 不迁移：18n μs
- 迁移：10 + 12n μs

当 18n > 10 + 12n，即 n > 1.67 时，迁移有利。

</details>

### 挑战题

**练习 15.4**：最优数据分区

给定一个 1TB 的张量和 4 个 NUMA 节点（每节点 256GB 内存），设计最优的数据分区方案。已知访问模式为：前 25% 数据访问频率是后 75% 的 4 倍。

*提示：考虑访问频率加权的负载均衡。*

<details>
<summary>答案</summary>

设前 25% 数据（256GB）的访问权重为 4，后 75% 数据（768GB）的访问权重为 1。

总访问权重 = 256GB × 4 + 768GB × 1 = 1024 + 768 = 1792

每节点应承担权重 = 1792 / 4 = 448

分区方案：
- 节点 0：前 112GB（权重 448）
- 节点 1：接下来 112GB（权重 448）  
- 节点 2：接下来 32GB（权重 128）+ 后部分 320GB（权重 320）
- 节点 3：最后 448GB（权重 448）

这样每个节点的访问负载基本均衡。

</details>

**练习 15.5**：Transformer 注意力机制的 NUMA 优化

对于一个序列长度 L=8192，隐藏维度 d=4096 的自注意力层，在 4 节点 NUMA 系统上如何分配 Q、K、V 矩阵以最小化通信开销？每个矩阵大小为 L×d×4 字节（float32）。

*提示：考虑注意力计算 $A = QK^T$ 的通信模式。*

<details>
<summary>答案</summary>

每个矩阵大小 = 8192 × 4096 × 4 = 128MB

策略 1：序列维度分区
- 将序列分成 4 份，每份 2048 个 token
- 节点 i 存储 Q[2048i:2048(i+1), :], K[2048i:2048(i+1), :], V[2048i:2048(i+1), :]
- 计算 $QK^T$ 时需要 all-to-all 通信
- 通信量：每节点发送 K 的 3/4 = 96MB

策略 2：头维度分区（假设多头注意力，h=32 头）
- 每节点处理 8 个注意力头
- 节点 i 存储所有序列的第 8i 到 8(i+1)-1 头
- 无需通信即可完成注意力计算
- 通信量：0（最优）

结论：头维度分区更优，实现了完全的 NUMA 局部性。

</details>

**练习 15.6**：动态负载均衡

设计一个 NUMA 感知的工作窃取算法，当某节点的任务队列为空时，如何决定从哪个节点窃取任务？给出决策函数。

*提示：考虑 NUMA 距离和队列长度的权衡。*

<details>
<summary>答案</summary>

决策函数：
$$\text{StealFrom}(i) = \arg\max_{j \neq i} \frac{Q_j - \tau}{d_{ij}}$$

其中：
- $Q_j$ 是节点 j 的队列长度
- $\tau$ 是窃取阈值（如 2）
- $d_{ij}$ 是 NUMA 距离

改进版本（考虑数据局部性）：
$$\text{StealFrom}(i) = \arg\max_{j \neq i} \frac{(Q_j - \tau) \cdot (1 + \lambda \cdot L_{ij})}{d_{ij}}$$

其中 $L_{ij}$ 是节点 j 的任务访问节点 i 数据的比例，$\lambda$ 是局部性权重。

算法步骤：
1. 计算所有非空节点的得分
2. 选择得分最高的节点
3. 窃取其队列尾部的任务（更可能有好的局部性）
4. 更新任务的数据亲和性信息

</details>

**练习 15.7**：内存分配器设计

设计一个 NUMA 感知的内存池，支持不同大小的内存块分配。要求：(1) 最小化跨节点分配，(2) 支持内存块迁移，(3) 避免碎片化。

*提示：使用分级内存池和迁移策略。*

<details>
<summary>答案</summary>

设计方案：

1. **分级结构**：
   - 小块池（<4KB）：每 CPU 核心私有
   - 中块池（4KB-1MB）：每 NUMA 节点共享
   - 大块池（>1MB）：全局池，NUMA 感知分配

2. **分配算法**：
```
Allocate(size, hint_node):
  if size < 4KB:
    return CPU_local_pool.alloc(size)
  elif size < 1MB:
    pool = NUMA_pools[hint_node]
    if pool.has_space(size):
      return pool.alloc(size)
    else:
      return find_nearest_pool(hint_node, size)
  else:
    return global_pool.alloc_numa_aware(size, hint_node)
```

3. **迁移策略**：
   - 监控跨节点访问计数器
   - 当 remote_access / total_access > 0.7 时触发迁移
   - 迁移时机：内存压力低时的后台任务

4. **防碎片化**：
   - 使用 buddy allocator 或 slab allocator
   - 定期整理：合并相邻空闲块
   - 大页面优先：优先分配 2MB 大页

</details>

**练习 15.8**：性能建模

建立一个 NUMA 系统的性能预测模型，输入：计算图、数据大小、NUMA 拓扑；输出：预期执行时间。考虑计算、内存访问和通信的重叠。

*提示：使用排队网络模型或 Roofline 模型的 NUMA 扩展。*

<details>
<summary>答案</summary>

性能模型：

$$T_{exec} = \max(T_{comp}, T_{mem}, T_{comm})$$

其中：

1. **计算时间**：
$$T_{comp} = \sum_{v \in V} \frac{FLOPs(v)}{throughput_{node(v)}}$$

2. **内存访问时间**：
$$T_{mem} = \sum_{v \in V} \left( \frac{M_{local}(v)}{BW_{local}} + \frac{M_{remote}(v)}{BW_{remote}} \right)$$

3. **通信时间**：
$$T_{comm} = \sum_{(u,v) \in E} \frac{data(u,v)}{BW_{interconnect}} \cdot overlap\_factor$$

**重叠因子**：
$$overlap\_factor = 1 - \min(\alpha_{comp-mem}, \alpha_{mem-comm})$$

其中 $\alpha$ 表示重叠程度，通过硬件特性和访问模式估算。

**NUMA 扩展的 Roofline 模型**：
$$Performance = \min\left(Peak\_FLOPs, \frac{AI \cdot BW_{eff}}{1 + \beta \cdot r_{remote}}\right)$$

其中：
- $AI$ 是算术强度
- $BW_{eff}$ 是有效带宽
- $r_{remote}$ 是远程访问比例
- $\beta$ 是 NUMA 惩罚系数

</details>

## 常见陷阱与错误 (Gotchas)

### 1. 首次触碰陷阱
**问题**：使用 `calloc` 或 `memset` 初始化大数组导致所有内存分配到单一节点。

**解决**：使用并行初始化，让每个线程初始化其将要访问的部分。

### 2. 错误的亲和性设置
**问题**：过度限制 CPU 亲和性导致负载不均衡。

**症状**：某些核心 100% 利用率，其他核心空闲。

**解决**：使用分层亲和性，允许在节点内迁移。

### 3. 页面抖动
**问题**：频繁的页面迁移导致性能下降。

**症状**：系统调用开销异常高。

**解决**：设置迁移阈值和冷却期。

### 4. 内存带宽瓶颈误判
**问题**：将 NUMA 远程访问延迟误认为是带宽不足。

**诊断**：使用硬件计数器区分延迟限制和带宽限制。

### 5. 不均匀的内存分配
**问题**：某些节点内存耗尽而其他节点有大量空闲。

**监控**：定期检查 `/proc/meminfo` 的每节点统计。

### 6. 忽略 NUMA 距离的非对称性
**问题**：假设 $d_{ij} = d_{ji}$，但某些系统中这不成立。

**验证**：始终检查完整的距离矩阵。

### 7. 缓存伪共享
**问题**：不同 NUMA 节点的线程访问同一缓存行的不同部分。

**解决**：使用填充（padding）或重新组织数据结构。

### 8. 大页面分配失败
**问题**：运行时无法分配大页面，回退到标准页面。

**预防**：启动时预留大页面，使用 `hugetlbfs`。

## 最佳实践检查清单

### 设计阶段
- [ ] 分析应用的内存访问模式和通信模式
- [ ] 评估数据并行 vs 模型并行的 NUMA 影响
- [ ] 设计支持 NUMA 的数据结构（避免全局共享状态）
- [ ] 规划内存容量需求，确保不超过节点容量

### 实现阶段
- [ ] 使用 NUMA 感知的内存分配器
- [ ] 实现分层的线程池和任务调度器
- [ ] 添加 NUMA 拓扑发现和自适应逻辑
- [ ] 实现关键数据结构的 NUMA 分区版本

### 优化阶段
- [ ] 使用硬件计数器监控远程访问比例
- [ ] 分析内存带宽利用率的均衡性
- [ ] 检查 TLB 命中率和大页面使用情况
- [ ] 验证 CPU 亲和性设置的有效性

### 测试阶段
- [ ] 在不同 NUMA 配置下测试（1/2/4/8 节点）
- [ ] 测试内存压力下的行为
- [ ] 验证故障转移和降级策略
- [ ] 基准测试：对比 NUMA 优化前后的性能

### 部署阶段
- [ ] 文档化 NUMA 相关的系统要求
- [ ] 提供 NUMA 配置的最佳实践指南
- [ ] 实现 NUMA 性能指标的监控和告警
- [ ] 准备 NUMA 相关问题的调试工具和流程

### 维护阶段
- [ ] 定期审查 NUMA 性能指标趋势
- [ ] 跟踪硬件拓扑变化（如 CPU 热插拔）
- [ ] 更新性能模型以反映实际运行数据
- [ ] 收集和分析 NUMA 相关的性能问题案例
