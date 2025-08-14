# 第 16 章：NUMA 架构优化（二）

本章深入探讨 NUMA 架构下的高级优化技术，重点关注跨 Socket 通信优化、NUMA 平衡算法、大规模 Transformer 模型的 NUMA 适配，以及性能分析调优方法。通过本章学习，读者将掌握在多 Socket 系统上部署 200T 规模模型的关键技术，理解如何最小化远程内存访问开销，实现近线性的扩展性。

## 16.1 跨 Socket 通信优化

### 16.1.1 通信拓扑分析

在 NUMA 系统中，跨 Socket 通信的拓扑结构直接影响性能。现代服务器通常采用以下几种互联方式：

```
2-Socket 全连接:
┌─────────┐  QPI/UPI  ┌─────────┐
│ Socket0 │◄──────────►│ Socket1 │
└─────────┘            └─────────┘

4-Socket 全互联:
┌─────────┐            ┌─────────┐
│ Socket0 │◄──────────►│ Socket1 │
└────┬────┘            └────┬────┘
     │      ╲        ╱      │
     │        ╲    ╱        │
     │          ╳           │
     │        ╱    ╲        │
     │      ╱        ╲      │
┌────▼────┐            ┌────▼────┐
│ Socket2 │◄──────────►│ Socket3 │
└─────────┘            └─────────┘

8-Socket 立方体拓扑:
         ┌───────┐
     ┌───│ S4    │───┐
     │   └───┬───┘   │
 ┌───▼───┐   │   ┌───▼───┐
 │ S0    │───┼───│ S5    │
 └───┬───┘   │   └───┬───┘
     │   ┌───▼───┐   │
     └───│ S1    │───┘
         └───┬───┘
             │
    [S2,S3,S6,S7 类似连接]
```

通信延迟矩阵可表示为：

$$L_{ij} = \begin{cases}
L_{local} & \text{if } i = j \\
L_{1hop} & \text{if 直接连接} \\
L_{2hop} & \text{if 需要一次中转} \\
L_{nhop} & \text{if 需要 n-1 次中转}
\end{cases}$$

其中典型值为：
- $L_{local} \approx 10ns$（本地内存访问）
- $L_{1hop} \approx 20-30ns$（直连 Socket）
- $L_{2hop} \approx 40-50ns$（一次中转）

### 16.1.2 通信模式优化

#### 点对点通信优化

对于大规模张量传输，采用分块流水线策略：

$$T_{total} = \max(T_{split}, T_{transfer}, T_{merge})$$

其中：
- $T_{split}$：数据切分时间
- $T_{transfer}$：传输时间，$T_{transfer} = \frac{S}{B_{inter}} + L_{ij}$
- $T_{merge}$：数据合并时间

优化策略：
1. **双缓冲机制**：传输与计算重叠
2. **NUMA 感知的数据分块**：块大小适配 LLC 容量
3. **传输聚合**：小消息合并减少开销

#### 集合通信优化

All-Reduce 操作的 NUMA 优化：

```
Ring All-Reduce with NUMA awareness:
Step 1: Intra-Socket Reduce
  Socket0: GPU0,1,2,3 → Local Reduce
  Socket1: GPU4,5,6,7 → Local Reduce

Step 2: Inter-Socket Exchange
  Socket0 ←→ Socket1 (仅交换一次)

Step 3: Intra-Socket Broadcast
  Socket0: Broadcast to GPU0,1,2,3
  Socket1: Broadcast to GPU4,5,6,7
```

通信复杂度分析：

$$T_{allreduce} = 2 \cdot \frac{(p-1) \cdot S}{p \cdot B_{intra}} + \frac{S}{B_{inter}}$$

其中：
- $p$：每个 Socket 内的 GPU 数量
- $S$：数据大小
- $B_{intra}$：Socket 内带宽
- $B_{inter}$：Socket 间带宽

### 16.1.3 内存一致性协议优化

NUMA 系统的缓存一致性协议（如 MESIF）开销分析：

状态转换成本矩阵：

$$C_{transition} = \begin{bmatrix}
0 & C_{M→E} & C_{M→S} & C_{M→I} & C_{M→F} \\
C_{E→M} & 0 & C_{E→S} & C_{E→I} & C_{E→F} \\
C_{S→M} & C_{S→E} & 0 & C_{S→I} & C_{S→F} \\
C_{I→M} & C_{I→E} & C_{I→S} & 0 & C_{I→F} \\
C_{F→M} & C_{F→E} & C_{F→S} & C_{F→I} & 0
\end{bmatrix}$$

优化原则：
1. **减少 False Sharing**：对齐到缓存行边界
2. **批量更新**：减少一致性协议触发次数
3. **只读数据复制**：利用 F 状态避免回写

## 16.2 NUMA 平衡算法

### 16.2.1 静态负载平衡

基于图分割的 NUMA 平衡算法：

给定计算图 $\mathcal{G} = (V, E)$，目标是找到 k-way 分割 $\Pi = \{V_1, V_2, ..., V_k\}$，最小化：

$$\min_{\Pi} \sum_{e=(u,v) \in E_{cut}} w(e) + \lambda \cdot \max_{i} \left(\sum_{v \in V_i} c(v)\right)$$

其中：
- $E_{cut}$：跨分区边集
- $w(e)$：边权重（通信量）
- $c(v)$：节点计算成本
- $\lambda$：平衡因子

使用多级图分割算法：
1. **粗化阶段**：合并相似节点
2. **初始分割**：谱聚类或 METIS
3. **细化阶段**：KL/FM 算法优化

### 16.2.2 动态负载迁移

运行时页面迁移策略：

$$M_{score}(p, n) = \alpha \cdot A_{local}(p, n) - \beta \cdot A_{remote}(p, n) - \gamma \cdot C_{migrate}$$

其中：
- $A_{local}(p, n)$：页面 p 在节点 n 的本地访问频率
- $A_{remote}(p, n)$：远程访问频率
- $C_{migrate}$：迁移成本

迁移决策阈值：

$$\text{Migrate if } M_{score}(p, n_{target}) - M_{score}(p, n_{current}) > \theta$$

### 16.2.3 自适应调度算法

基于强化学习的 NUMA 调度器：

状态空间 $\mathcal{S}$：
- 各节点内存使用率：$\{m_1, m_2, ..., m_k\}$
- 跨节点通信矩阵：$\mathbf{C} \in \mathbb{R}^{k \times k}$
- 任务队列长度：$\{q_1, q_2, ..., q_k\}$

动作空间 $\mathcal{A}$：
- 任务分配：$a_{task} \in \{1, 2, ..., k\}$
- 数据放置：$a_{data} \in \{1, 2, ..., k\}$

奖励函数：

$$R = -\left(T_{exec} + \alpha \cdot T_{comm} + \beta \cdot \text{Imbalance}\right)$$

## 16.3 大规模 Transformer 的 NUMA 优化

### 16.3.1 注意力机制的 NUMA 分解

对于序列长度 $L$、隐藏维度 $d$、头数 $h$ 的多头注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

NUMA 分解策略：

1. **序列维度分割**：
   - Socket $i$ 处理序列片段 $[L_i, L_{i+1})$
   - 通信量：$O(h \cdot d \cdot L)$

2. **注意力头分割**：
   - Socket $i$ 处理头 $[h_i, h_{i+1})$
   - 通信量：$O(L^2)$（仅 softmax 归一化）

3. **混合分割**：
   $$\text{Socket}(i,j) = \text{Heads}[ih:ih+\Delta h] \times \text{Seq}[jL:jL+\Delta L]$$

内存访问模式优化：

```
Blocked Attention Computation:
┌────────────────────────┐
│  Q blocks (Socket 0)   │
├────┬────┬────┬────────┤
│ B00│ B01│ B02│  ...   │
├────┼────┼────┼────────┤
│ K,V blocks distributed │
│   across Sockets       │
└────────────────────────┘
```

### 16.3.2 FFN 层的 NUMA 优化

前馈网络层：

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

其中 $W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$

优化策略：
1. **列并行分割** $W_1$：
   $$W_1 = [W_1^{(0)} | W_1^{(1)} | ... | W_1^{(k-1)}]$$
   每个 Socket 计算部分激活

2. **行并行分割** $W_2$：
   $$W_2 = \begin{bmatrix} W_2^{(0)} \\ W_2^{(1)} \\ \vdots \\ W_2^{(k-1)} \end{bmatrix}$$
   需要 All-Reduce 合并结果

通信成本分析：
- 列并行：无通信（计算独立）
- 行并行：$O(batch \times seq \times d)$ 的 All-Reduce

### 16.3.3 KV Cache 的 NUMA 管理

对于长序列生成任务，KV Cache 的 NUMA 布局至关重要：

内存占用：
$$M_{KV} = 2 \times layers \times heads \times seq\_len \times d_k \times batch$$

分布式 KV Cache 设计：

```
NUMA-aware KV Cache Layout:
┌─────────────────────────────┐
│ Socket 0: Layers 0-15       │
│ ┌─────────┬───────────────┐ │
│ │ K-Cache │ Head 0-3      │ │
│ ├─────────┼───────────────┤ │
│ │ V-Cache │ Head 0-3      │ │
│ └─────────┴───────────────┘ │
├─────────────────────────────┤
│ Socket 1: Layers 16-31      │
│ [Similar structure]         │
└─────────────────────────────┘
```

访问模式优化：
1. **预取策略**：基于生成位置的预测性预取
2. **层间复用**：相邻层共享 Socket 减少迁移
3. **压缩存储**：量化 + 稀疏表示降低带宽需求

### 16.3.4 Pipeline 并行的 NUMA 适配

流水线并行中的 NUMA 感知调度：

$$\text{Stage}_{assignment} = \arg\min_{S} \left( T_{compute}^{max} + T_{comm}^{total} \right)$$

约束条件：
- 内存约束：$\sum_{s \in S_i} M(s) \leq M_{node_i}$
- 带宽约束：$B_{used} \leq B_{available}$

优化的微批次调度：

```
1F1B Schedule with NUMA:
Time →
Socket0: F0 F1 F2 F3 B3 B2 B1 B0 F4 F5...
Socket1:    F0 F1 F2 F3 B3 B2 B1 B0 F4...
Socket2:       F0 F1 F2 F3 B3 B2 B1 B0...
Socket3:          F0 F1 F2 F3 B3 B2 B1...

F=Forward, B=Backward
数字表示微批次编号
```

## 16.4 性能分析与调优

### 16.4.1 性能指标体系

NUMA 性能指标层次：

1. **硬件层指标**：
   - 本地/远程内存访问比：$R_{L/R} = \frac{N_{local}}{N_{remote}}$
   - QPI/UPI 利用率：$U_{link} = \frac{B_{actual}}{B_{theoretical}}$
   - 内存控制器饱和度：$S_{MC} = \frac{Requests_{actual}}{Requests_{max}}$

2. **系统层指标**：
   - NUMA 节点负载均衡度：$B_{load} = 1 - \frac{\sigma(Load_i)}{\mu(Load_i)}$
   - 页面迁移频率：$F_{migrate} = \frac{Pages_{migrated}}{Time}$
   - 缓存一致性开销：$O_{coherence} = \frac{T_{coherence}}{T_{total}}$

3. **应用层指标**：
   - 计算通信比：$\rho = \frac{T_{compute}}{T_{communication}}$
   - 并行效率：$E = \frac{Speedup}{N_{sockets}}$
   - 内存带宽效率：$\eta_{mem} = \frac{BW_{achieved}}{BW_{peak} \times N_{sockets}}$

### 16.4.2 性能分析工具

使用 PMU 计数器进行细粒度分析：

```
关键性能事件：
- UNC_M_CAS_COUNT.RD: 内存读请求
- UNC_M_CAS_COUNT.WR: 内存写请求  
- UNC_Q_RxL_FLITS: QPI/UPI 流量
- OFFCORE_RESPONSE: 远程内存访问
```

性能瓶颈识别决策树：

```
                高延迟？
                   │
         ┌─────────┴─────────┐
         │                   │
         是                  否
         │                   │
    远程访问多？          带宽饱和？
         │                   │
    ┌────┴────┐         ┌────┴────┐
    是        否        是        否
    │         │         │         │
数据布局  缓存竞争  通信优化  计算优化
```

### 16.4.3 调优策略矩阵

基于 Roofline 模型的 NUMA 调优：

$$P = \min\left(P_{peak}, I \times BW_{effective}\right)$$

其中有效带宽考虑 NUMA 因素：

$$BW_{effective} = \alpha \cdot BW_{local} + (1-\alpha) \cdot BW_{remote}$$

$\alpha$ 是本地访问比例。

调优决策表：

| 算术强度 | NUMA 比例 | 优化策略 |
|---------|----------|---------|
| 低 (<0.5) | 高远程 | 数据重分布 |
| 低 (<0.5) | 高本地 | 预取优化 |
| 中 (0.5-4) | 高远程 | 计算迁移 |
| 中 (0.5-4) | 高本地 | 向量化 |
| 高 (>4) | 任意 | 计算优化 |

### 16.4.4 案例：200T 模型的 NUMA 调优

实际 200T 参数模型在 8-Socket 系统上的优化过程：

**初始性能分析**：
- 远程内存访问：45%
- QPI 带宽利用率：78%
- 并行效率：4.2/8 = 52.5%

**优化步骤**：

1. **重新设计数据布局**：
   - 权重矩阵按 Socket 边界对齐
   - 激活张量采用 NUMA-aware allocation
   - 结果：远程访问降至 28%

2. **优化通信模式**：
   - Ring AllReduce → Hierarchical AllReduce
   - 小消息聚合，批大小从 1MB → 16MB
   - 结果：QPI 利用率降至 65%

3. **调整并行策略**：
   - 从纯数据并行改为混合并行
   - Pipeline stages 与 Socket 边界对齐
   - 结果：并行效率提升至 6.8/8 = 85%

**最终性能提升**：
- 端到端吞吐量：2.4x
- 内存带宽利用率：82%
- 延迟降低：35%

性能剖析对比：

```
优化前时间分布：          优化后时间分布：
┌───────────────┐        ┌───────────────┐
│ Compute: 45%  │        │ Compute: 68%  │
├───────────────┤        ├───────────────┤
│ Local Mem: 20%│        │ Local Mem: 22%│
├───────────────┤        ├───────────────┤
│ Remote Mem:25%│        │ Remote Mem: 7%│
├───────────────┤        ├───────────────┤
│ Sync: 10%     │        │ Sync: 3%      │
└───────────────┘        └───────────────┘
```

## 本章小结

本章深入探讨了 NUMA 架构的高级优化技术，涵盖了从硬件层面的跨 Socket 通信优化到应用层面的大规模 Transformer 模型适配。核心要点包括：

1. **跨 Socket 通信优化**：理解不同拓扑结构的通信特性，采用分块流水线、双缓冲、NUMA 感知的集合通信等策略，显著降低远程访问开销。

2. **NUMA 平衡算法**：通过静态图分割、动态页面迁移和自适应调度，实现计算和数据的最优分布，关键公式：
   - 迁移评分：$M_{score}(p, n) = \alpha \cdot A_{local} - \beta \cdot A_{remote} - \gamma \cdot C_{migrate}$
   - 图分割目标：$\min \sum_{e \in E_{cut}} w(e) + \lambda \cdot \max_i \sum_{v \in V_i} c(v)$

3. **Transformer NUMA 优化**：针对注意力机制、FFN 层、KV Cache 的特点设计专门的 NUMA 分解策略，实现近线性扩展。

4. **性能分析与调优**：建立多层次性能指标体系，使用 PMU 计数器精确定位瓶颈，通过实际案例展示 2.4x 的性能提升。

关键性能公式汇总：
- 有效带宽：$BW_{effective} = \alpha \cdot BW_{local} + (1-\alpha) \cdot BW_{remote}$
- 并行效率：$E = \frac{Speedup}{N_{sockets}}$
- Roofline 性能：$P = \min(P_{peak}, I \times BW_{effective})$

## 练习题

### 基础题

1. **通信延迟计算**
   在一个 4-Socket 全互联系统中，本地内存访问延迟为 10ns，直连 Socket 访问延迟为 25ns，需要一次中转的访问延迟为 45ns。如果一个应用有 60% 本地访问、30% 直连访问、10% 需要中转的访问，计算平均内存访问延迟。

   <details>
   <summary>答案</summary>
   
   平均延迟 = 0.6 × 10 + 0.3 × 25 + 0.1 × 45 = 6 + 7.5 + 4.5 = 18ns
   
   相比纯本地访问，延迟增加了 80%，这说明 NUMA 优化的重要性。
   </details>

2. **All-Reduce 通信量分析**
   在 8 个 GPU（分布在 2 个 Socket，每个 4 GPU）的系统上执行 Ring All-Reduce，数据大小为 4GB。如果 Socket 内带宽为 200GB/s，Socket 间带宽为 50GB/s，计算 NUMA 感知的分层 All-Reduce 相比普通 Ring All-Reduce 的加速比。

   <details>
   <summary>答案</summary>
   
   普通 Ring All-Reduce：
   - 总传输量：7 × 4GB = 28GB
   - 假设一半通过 Socket 间链路：14GB / 50GB/s = 0.28s
   
   NUMA 分层 All-Reduce：
   - Socket 内 Reduce：3 × 4GB / 200GB/s × 2 = 0.06s
   - Socket 间交换：4GB / 50GB/s = 0.08s  
   - Socket 内 Broadcast：3 × 4GB / 200GB/s × 2 = 0.06s
   - 总时间：max(0.06, 0.08, 0.06) = 0.08s（并行执行）
   
   加速比：0.28 / 0.08 = 3.5x
   </details>

3. **内存带宽效率计算**
   一个 NUMA 系统有 4 个节点，每个节点峰值带宽 100GB/s。测量得到本地访问带宽 90GB/s，远程访问带宽 30GB/s。如果应用的本地访问比例为 75%，计算内存带宽效率。

   <details>
   <summary>答案</summary>
   
   有效带宽 = 0.75 × 90 + 0.25 × 30 = 67.5 + 7.5 = 75GB/s
   
   总峰值带宽 = 4 × 100 = 400GB/s
   实际总带宽 = 4 × 75 = 300GB/s（假设均衡负载）
   
   带宽效率 = 300 / 400 = 75%
   </details>

### 挑战题

4. **NUMA 感知的矩阵乘法分块**
   设计一个 NUMA 感知的矩阵乘法 C = A × B 的分块策略，其中 A 是 M×K 矩阵，B 是 K×N 矩阵，系统有 P 个 NUMA 节点。要求最小化跨节点通信量，给出分块大小和数据分布方案。

   **Hint**: 考虑 2D 分块和 Cannon 算法的 NUMA 适配。

   <details>
   <summary>答案</summary>
   
   采用 2D 分块，将处理器组织为 √P × √P 网格：
   
   1. 分块大小：
      - A 分块：(M/√P) × K
      - B 分块：K × (N/√P)  
      - C 分块：(M/√P) × (N/√P)
   
   2. 初始数据分布：
      - 节点 (i,j) 持有 A[i,:] 和 B[:,j]
   
   3. 计算步骤（Cannon 算法）：
      - 初始对齐：A 的第 i 行循环左移 i 位，B 的第 j 列循环上移 j 位
      - √P 轮迭代，每轮：
        * 本地计算：C[i,j] += A[i,k] × B[k,j]
        * A 左移一位，B 上移一位
   
   4. 通信量分析：
      - 每个节点每轮发送：MK/P + KN/P
      - 总通信量：√P × (MK + KN) / √P = MK + KN
      - 相比朴素方法减少 √P 倍
   </details>

5. **动态 NUMA 负载平衡算法**
   设计一个基于工作窃取的 NUMA 感知调度器，考虑任务亲和性和数据局部性。给出窃取策略和性能模型。

   **Hint**: 分层工作窃取，优先从同一 NUMA 节点窃取。

   <details>
   <summary>答案</summary>
   
   分层工作窃取算法：
   
   1. 队列组织：
      - 每个核心维护本地队列
      - 每个 NUMA 节点维护共享队列
      - 全局队列作为最后手段
   
   2. 窃取优先级：
      - Level 0：本地队列（无开销）
      - Level 1：同一 NUMA 节点其他核心（低开销）
      - Level 2：直连 NUMA 节点（中等开销）
      - Level 3：远程 NUMA 节点（高开销）
   
   3. 窃取决策函数：
      $$Steal(i,j) = \begin{cases}
      1 & \text{if } Q_j > \theta_1 \text{ and } d(i,j) = 0 \\
      p_1 & \text{if } Q_j > \theta_2 \text{ and } d(i,j) = 1 \\
      p_2 & \text{if } Q_j > \theta_3 \text{ and } d(i,j) = 2 \\
      0 & \text{otherwise}
      \end{cases}$$
   
   4. 性能模型：
      - 负载不均衡成本：$C_{imbalance} = \sigma(Q_i) \times T_{task}$
      - 窃取开销：$C_{steal} = \sum_{i,j} N_{steal}(i,j) \times L_{ij}$
      - 总成本：$C_{total} = C_{imbalance} + C_{steal}$
   
   5. 自适应参数调整：
      - 监控窃取成功率
      - 动态调整阈值 θ 和概率 p
      - 使用指数加权移动平均平滑
   </details>

6. **200T 模型的 NUMA 内存规划**
   一个 200T 参数的 GPT 模型需要在 8-Socket 系统上部署，每个 Socket 有 1TB 内存。设计内存布局方案，考虑模型并行、数据并行和流水线并行的混合策略。

   **Hint**: 考虑参数、激活、优化器状态和 KV Cache 的分布。

   <details>
   <summary>答案</summary>
   
   内存规划方案：
   
   1. 内存需求分析（FP16 + 混合精度训练）：
      - 模型参数：200T × 2B = 400TB
      - 梯度：200T × 2B = 400TB  
      - 优化器状态（Adam）：200T × 8B = 1600TB
      - 激活值（seq=8K, batch=512）：约 100TB
      - KV Cache（生成）：约 50TB
      - 总需求：约 2550TB
   
   2. 并行策略（8 Sockets × 1TB = 8TB 可用）：
      - 模型并行度：MP = 64
      - 数据并行度：DP = 32  
      - 流水线并行度：PP = 8
      - 总并行度：64 × 32 × 8 = 16384
   
   3. Socket 级别分配：
      - 每个 Socket 负责 PP 的一个 stage
      - Stage 内部做 MP，每 Socket 8 个 MP 分片
      - 参数：400TB / 16384 ≈ 25GB per rank
      - 优化器：1600TB / 16384 ≈ 100GB per rank
   
   4. NUMA 优化布局：
      ```
      Socket 0-1: Layers 0-31 (Embedding + Early layers)
      Socket 2-3: Layers 32-63 
      Socket 4-5: Layers 64-95
      Socket 6-7: Layers 96-127 (Output + Late layers)
      ```
   
   5. 数据流优化：
      - Forward：Socket 0→1→...→7
      - Backward：Socket 7→6→...→0
      - 使用双缓冲隐藏通信延迟
      - 激活检查点存储在产生它的 Socket
   
   6. 内存复用策略：
      - 激活值在 backward 后立即释放
      - KV Cache 使用环形缓冲区
      - 优化器状态分片存储，按需换入
   </details>

## 常见陷阱与错误 (Gotchas)

1. **False Sharing 导致的性能退化**
   - 错误：多个线程更新同一缓存行的不同变量
   - 解决：使用 padding 对齐到缓存行边界（通常 64 字节）

2. **不当的内存分配策略**
   - 错误：使用默认 malloc，导致页面分配到错误的 NUMA 节点
   - 解决：使用 numa_alloc_onnode 或设置内存策略

3. **忽视 NUMA 距离的影响**
   - 错误：假设所有远程访问代价相同
   - 解决：查询 NUMA 距离矩阵，考虑多跳访问

4. **过度的页面迁移**
   - 错误：频繁迁移页面导致开销超过收益
   - 解决：设置迁移阈值，使用迁移批处理

5. **线程绑定不当**
   - 错误：线程在 NUMA 节点间迁移
   - 解决：使用 CPU affinity 绑定线程到特定核心

6. **忽略 QPI/UPI 带宽限制**
   - 错误：过度依赖跨 Socket 通信
   - 解决：监控链路利用率，调整通信模式

## 最佳实践检查清单

### 设计阶段
- [ ] 分析目标硬件的 NUMA 拓扑结构
- [ ] 评估应用的内存访问模式和通信需求
- [ ] 选择合适的并行策略（数据/模型/流水线）
- [ ] 设计 NUMA 感知的数据结构和算法
- [ ] 规划内存布局，最小化跨节点访问

### 实现阶段
- [ ] 使用 NUMA API 进行内存分配
- [ ] 实现分层通信原语
- [ ] 添加线程/进程绑定
- [ ] 实现数据预取和缓存优化
- [ ] 使用内存池减少分配开销

### 调优阶段
- [ ] 使用 PMU 计数器收集性能数据
- [ ] 分析本地/远程访问比例
- [ ] 监控 QPI/UPI 链路利用率
- [ ] 评估负载均衡效果
- [ ] 迭代优化热点代码路径

### 验证阶段
- [ ] 在不同 NUMA 配置下测试
- [ ] 验证扩展性（weak/strong scaling）
- [ ] 检查内存泄漏和碎片化
- [ ] 确认性能指标达到预期
- [ ] 文档化 NUMA 相关配置要求