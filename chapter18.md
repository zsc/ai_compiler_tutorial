# 第 18 章：动态 Shape 编译（二）

在上一章中，我们探讨了动态 shape 的静态分析技术，包括符号形状推导、约束求解和桶化策略。本章将深入运行时层面，研究如何通过运行时特化、智能重编译和缓存机制，在保持灵活性的同时实现接近静态 shape 的性能。这些技术对于自动驾驶中的可变目标检测和具身智能的动态环境感知至关重要。

## 18.1 运行时特化

运行时特化是动态 shape 编译的核心技术，通过在运行时根据实际形状生成优化代码，实现性能与灵活性的平衡。

### 18.1.1 特化时机决策

特化时机的选择直接影响系统性能。过早特化会增加编译开销，过晚特化则错失优化机会。

**热度阈值模型**：

设形状 $s$ 在时间窗口 $[t-w, t]$ 内出现次数为 $f(s,t,w)$，特化触发条件为：

$$\begin{cases}
\text{specialize}(s) & \text{if } f(s,t,w) > \theta_{\text{hot}} \\
\text{defer}(s) & \text{if } \theta_{\text{cold}} < f(s,t,w) \leq \theta_{\text{hot}} \\
\text{interpret}(s) & \text{if } f(s,t,w) \leq \theta_{\text{cold}}
\end{cases}$$

其中 $\theta_{\text{hot}}$ 和 $\theta_{\text{cold}}$ 是动态调整的阈值。

**收益预测函数**：

$$B(s) = (T_{\text{interp}}(s) - T_{\text{spec}}(s)) \times P_{\text{reuse}}(s) - C_{\text{compile}}(s)$$

其中：
- $T_{\text{interp}}(s)$：解释执行时间
- $T_{\text{spec}}(s)$：特化代码执行时间  
- $P_{\text{reuse}}(s)$：形状重用概率
- $C_{\text{compile}}(s)$：编译开销

### 18.1.2 代码生成策略

运行时代码生成需要在编译质量和编译速度之间权衡。

**分层特化架构**：

```
Level 0: 通用解释器
   ↓ (threshold: 10 calls)
Level 1: 基础特化（循环展开、边界检查消除）
   ↓ (threshold: 100 calls)
Level 2: 深度优化（向量化、预取优化）
   ↓ (threshold: 1000 calls)
Level 3: 极致优化（自动调优、硬件特定优化）
```

**特化代码模板**：

对于卷积操作 $Y = X * W$，其中 $X \in \mathbb{R}^{N \times C \times H \times W}$：

通用模板：
$$Y_{n,k,i,j} = \sum_{c,u,v} X_{n,c,i+u,j+v} \cdot W_{k,c,u,v}$$

特化实例（当 $H=224, W=224$ 时）：
- 消除边界检查
- 循环分块对齐缓存行
- 向量化内层循环

### 18.1.3 内存管理优化

动态 shape 的内存管理面临碎片化和分配开销问题。

**内存池设计**：

采用分级内存池管理：

$$\text{Pool}_i = \{b | 2^i \leq \text{size}(b) < 2^{i+1}\}$$

分配策略：
$$\text{allocate}(s) = \begin{cases}
\text{Pool}_{\lceil \log_2(s) \rceil} & \text{if available} \\
\text{malloc}(s) & \text{otherwise}
\end{cases}$$

**预分配策略**：

基于历史模式预测未来内存需求：

$$M_{\text{pred}}(t+1) = \alpha \cdot M_{\text{obs}}(t) + (1-\alpha) \cdot M_{\text{pred}}(t)$$

其中 $\alpha$ 是平滑因子，通常取 $0.3 \sim 0.5$。

## 18.2 重编译触发机制

智能的重编译触发机制是平衡性能和资源消耗的关键。

### 18.2.1 触发条件设计

**多维触发条件**：

定义触发向量 $\vec{T} = (T_{\text{freq}}, T_{\text{perf}}, T_{\text{mem}}, T_{\text{shape}})$：

- $T_{\text{freq}}$：执行频率触发
- $T_{\text{perf}}$：性能退化触发
- $T_{\text{mem}}$：内存压力触发
- $T_{\text{shape}}$：形状变化触发

触发决策函数：
$$\text{trigger} = \bigvee_{i} (T_i > \theta_i) \vee \left(\sum_{i} w_i \cdot T_i > \theta_{\text{global}}\right)$$

### 18.2.2 阈值自适应调整

**指数移动平均调整**：

$$\theta_{\text{new}} = \begin{cases}
\theta_{\text{old}} \times (1 + \beta) & \text{if } \text{benefit} > \text{cost} \\
\theta_{\text{old}} \times (1 - \beta) & \text{otherwise}
\end{cases}$$

其中 $\beta$ 是调整步长，通常取 $0.1$。

**基于强化学习的阈值优化**：

将阈值调整建模为马尔可夫决策过程（MDP）：

- 状态：$s = (\text{workload}, \text{resources}, \text{history})$
- 动作：$a = \Delta\theta$
- 奖励：$r = \text{speedup} - \lambda \cdot \text{compile\_cost}$

使用 Q-learning 更新策略：
$$Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a')]$$

### 18.2.3 热点检测算法

**滑动窗口热点检测**：

维护时间窗口 $W = [t-\tau, t]$ 内的执行统计：

$$H(op, s) = \frac{\sum_{i \in W} \mathbb{1}[\text{shape}_i = s] \cdot \text{time}_i}{\sum_{i \in W} \text{time}_i}$$

当 $H(op, s) > \theta_{\text{hot}}$ 时，标记为热点。

**自适应采样**：

使用 reservoir sampling 降低统计开销：

$$P_{\text{sample}} = \min\left(1, \frac{k}{\text{count}}\right)$$

其中 $k$ 是 reservoir 大小。

## 18.3 Shape 缓存策略

有效的缓存策略可以显著减少重编译开销，提高动态 shape 系统的整体性能。

### 18.3.1 缓存结构设计

**多级缓存架构**：

```
L1 Cache (Per-Op)
├── Exact Match Cache
│   └── Hash Table: shape → compiled_code
├── Pattern Cache  
│   └── Trie: shape_pattern → template_code
└── Range Cache
    └── Interval Tree: shape_range → parametric_code

L2 Cache (Global)
├── Shared Code Cache
└── Cross-Op Optimization Cache
```

**缓存键设计**：

对于形状 $s = (d_1, d_2, ..., d_n)$，缓存键计算：

$$\text{key}(s) = \text{hash}(s) \oplus \text{hash}(\text{dtype}) \oplus \text{hash}(\text{layout})$$

引入局部敏感哈希（LSH）支持相似形状查找：

$$\text{LSH}(s) = \left\lfloor \frac{s}{g} \right\rfloor$$

其中 $g$ 是粒度参数。

**版本管理**：

每个缓存项包含版本信息：

$$\text{entry} = \{\text{code}, \text{version}, \text{deps}, \text{stats}\}$$

版本兼容性检查：
$$\text{compatible}(v_1, v_2) = \bigwedge_{d \in \text{deps}} (\text{version}(d) = \text{version}_{\text{cached}}(d))$$

### 18.3.2 替换算法

**自适应 LRU-K**：

结合访问频率和最近性：

$$\text{priority}(e) = \sum_{i=1}^{K} w_i \cdot \frac{1}{t - t_i(e)}$$

其中 $t_i(e)$ 是倒数第 $i$ 次访问时间。

**成本感知替换**：

考虑编译成本的替换决策：

$$\text{evict\_score}(e) = \frac{\text{age}(e) \cdot \text{size}(e)}{\text{freq}(e) \cdot \text{compile\_cost}(e)}$$

选择 score 最高的项进行替换。

**预测性预取**：

基于形状序列预测：

$$P(s_{t+1} | s_t, s_{t-1}, ...) = \frac{\text{count}(s_t \rightarrow s_{t+1})}{\text{count}(s_t)}$$

当 $P(s') > \theta_{\text{prefetch}}$ 时，预编译形状 $s'$。

### 18.3.3 缓存有效性验证

**增量验证**：

$$\text{valid}(c) = \text{checksum}(c) = \text{expected} \wedge \forall d \in \text{deps}(c): \text{unchanged}(d)$$

**依赖追踪图**：

构建缓存项之间的依赖关系：

```
    Op1_Cache
    /        \
   v          v
Op2_Cache  Op3_Cache
   \          /
    v        v
    Op4_Cache
```

当上游缓存失效时，递归标记下游缓存。

**一致性协议**：

在分布式环境中，使用两阶段提交保证缓存一致性：

1. Prepare: $\forall n \in \text{nodes}: \text{lock}(n, \text{key})$
2. Commit: $\forall n \in \text{nodes}: \text{update}(n, \text{key}, \text{value})$

## 18.4 性能预测模型

准确的性能预测是优化决策的基础。

### 18.4.1 成本模型构建

**分层成本模型**：

$$C_{\text{total}} = C_{\text{compute}} + C_{\text{memory}} + C_{\text{comm}} + C_{\text{overhead}}$$

计算成本：
$$C_{\text{compute}} = \sum_{op} \text{FLOPS}(op) \times \frac{1}{\text{efficiency}(op, \text{hw})}$$

内存成本：
$$C_{\text{memory}} = \sum_{t} \frac{\text{size}(t)}{\text{BW}_{\text{level}(t)}} + \text{miss\_penalty} \times P_{\text{miss}}$$

通信成本（for 分布式）：
$$C_{\text{comm}} = \alpha + \beta \times \text{msg\_size} + \gamma \times \text{hop\_count}$$

### 18.4.2 机器学习辅助预测

**特征工程**：

提取关键特征向量 $\vec{f}$：
- 形状特征：$(N, C, H, W, \text{aspect\_ratio}, \text{volume})$
- 算子特征：$(\text{type}, \text{params}, \text{memory\_pattern})$
- 硬件特征：$(\text{compute\_units}, \text{memory\_bw}, \text{cache\_size})$

**XGBoost 预测模型**：

$$T_{\text{pred}} = \sum_{k=1}^{K} f_k(\vec{f})$$

其中 $f_k$ 是第 $k$ 棵决策树。

训练目标：
$$\mathcal{L} = \sum_{i} (T_{\text{pred}}^{(i)} - T_{\text{actual}}^{(i)})^2 + \sum_{k} \Omega(f_k)$$

**在线学习更新**：

使用指数加权移动平均更新模型参数：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t, x_t, y_t)$$

### 18.4.3 自适应调整机制

**贝叶斯优化**：

使用高斯过程建模性能函数：

$$f \sim \mathcal{GP}(\mu, k)$$

采集函数（Expected Improvement）：

$$\text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)]$$

其中 $f^*$ 是当前最优值。

**多臂老虎机策略**：

平衡探索与利用：

$$a_t = \arg\max_a \left[ \hat{\mu}_a + \sqrt{\frac{2\ln t}{n_a}} \right]$$

其中 $\hat{\mu}_a$ 是动作 $a$ 的平均收益，$n_a$ 是选择次数。

**反馈控制环**：

```
目标性能 → PID控制器 → 参数调整
    ↑                      ↓
实际性能 ← 系统执行 ← 新参数
```

PID 控制器：
$$u(t) = K_p e(t) + K_i \int_0^t e(\tau)d\tau + K_d \frac{de(t)}{dt}$$

## 本章小结

本章深入探讨了动态 shape 编译的运行时技术，涵盖了从特化决策到性能预测的完整体系：

1. **运行时特化**：通过热度阈值模型和分层特化架构，在编译开销和执行效率之间找到平衡点。关键公式：$B(s) = (T_{\text{interp}}(s) - T_{\text{spec}}(s)) \times P_{\text{reuse}}(s) - C_{\text{compile}}(s)$

2. **重编译触发**：设计了多维触发条件和自适应阈值调整机制，使用强化学习优化触发策略。核心是平衡探索与利用的权衡。

3. **缓存策略**：构建了多级缓存架构，结合 LRU-K 和成本感知的替换算法，通过局部敏感哈希支持相似形状查找。

4. **性能预测**：建立了分层成本模型 $C_{\text{total}} = C_{\text{compute}} + C_{\text{memory}} + C_{\text{comm}} + C_{\text{overhead}}$，并使用机器学习方法提高预测准确性。

这些技术的综合应用，使得动态 shape 系统能够在保持灵活性的同时，达到接近静态 shape 系统 85-95% 的性能水平。

## 练习题

### 基础题

**练习 18.1**：给定一个形状序列 $S = [(32,3,224,224), (64,3,224,224), (32,3,224,224), (128,3,224,224), (32,3,224,224)]$，热度阈值 $\theta_{\text{hot}} = 2$，计算哪些形状应该被特化？

*Hint*：统计每个形状的出现频率，与阈值比较。

<details>
<summary>答案</summary>

形状 $(32,3,224,224)$ 出现 3 次，超过阈值 2，应该被特化。
其他形状各出现 1 次，不应被特化。

</details>

**练习 18.2**：一个算子的解释执行时间为 100ms，特化后执行时间为 20ms，编译开销为 500ms。如果形状重用概率为 0.8，计算特化收益 $B(s)$。

*Hint*：直接代入收益预测公式。

<details>
<summary>答案</summary>

$$B(s) = (100 - 20) \times 0.8 - 500 = 80 \times 0.8 - 500 = 64 - 500 = -436\text{ms}$$

收益为负，不应该特化。若要使收益为正，需要形状重用概率 > 6.25。

</details>

**练习 18.3**：设计一个简单的 LRU-2 缓存替换算法，缓存容量为 3，处理访问序列：A, B, C, D, B, A, E, B, C。列出每步后的缓存状态。

*Hint*：LRU-2 考虑倒数第二次访问时间。

<details>
<summary>答案</summary>

步骤追踪：
1. A → [A]
2. B → [A, B]
3. C → [A, B, C]
4. D → [B, C, D] (淘汰 A，因为 A 只访问过一次)
5. B → [B, C, D] (B 更新)
6. A → [B, C, A] (淘汰 D)
7. E → [B, A, E] (淘汰 C，C 的倒数第二次访问最早)
8. B → [B, A, E] (B 更新)
9. C → [B, A, C] (淘汰 E)

最终缓存：[B, A, C]

</details>

### 挑战题

**练习 18.4**：设计一个自适应阈值调整算法。已知最近 10 次重编译的收益/成本比为：[0.5, 0.8, 1.2, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4, 1.6]。初始阈值 $\theta = 100$，调整步长 $\beta = 0.1$。计算最终阈值。

*Hint*：根据收益/成本比决定调整方向。

<details>
<summary>答案</summary>

分析每次调整：
- 比值 > 1 的次数：6 次（上调）
- 比值 ≤ 1 的次数：4 次（下调）

累积调整：
$$\theta_{\text{final}} = 100 \times (1.1)^6 \times (0.9)^4 = 100 \times 1.772 \times 0.656 = 116.2$$

考虑到实际中可能使用移动平均或其他平滑策略，最终阈值约为 116。

</details>

**练习 18.5**：给定一个具有以下特性的深度学习模型：
- 输入形状在 $[1, 32] \times 3 \times [224, 512] \times [224, 512]$ 范围内变化
- batch size 以概率 0.7 为 1，0.2 为 8，0.1 为 32
- 图像尺寸以概率 0.8 为 224×224，0.2 为 512×512

设计一个桶化策略，使得缓存命中率最大化，同时限制桶数量不超过 6。

*Hint*：考虑概率分布和性能影响。

<details>
<summary>答案</summary>

优化策略：
1. Batch 维度：[1], [8], [32] - 3 个桶
2. 空间维度：[224×224], [512×512] - 2 个桶

组合后共 3×2 = 6 个桶：
- (1, 224×224) - 概率 0.7×0.8 = 0.56
- (1, 512×512) - 概率 0.7×0.2 = 0.14
- (8, 224×224) - 概率 0.2×0.8 = 0.16
- (8, 512×512) - 概率 0.2×0.2 = 0.04
- (32, 224×224) - 概率 0.1×0.8 = 0.08
- (32, 512×512) - 概率 0.1×0.2 = 0.02

理论缓存命中率 = 100%（所有情况都被覆盖）

</details>

**练习 18.6**：使用贝叶斯优化选择最优编译参数。已知性能函数的三个观测点：
- $x_1 = 0.2$, $y_1 = 0.6$
- $x_2 = 0.5$, $y_2 = 0.8$  
- $x_3 = 0.8$, $y_3 = 0.7$

假设高斯过程的均值为 0，核函数为 RBF。计算 $x = 0.6$ 处的期望改进（EI）。

*Hint*：先估计该点的均值和方差。

<details>
<summary>答案</summary>

使用 RBF 核函数插值：
1. 计算协方差矩阵 K
2. 预测 $x = 0.6$ 的均值：约 0.82
3. 预测方差：约 0.05
4. 当前最优 $f^* = 0.8$

期望改进：
$$\text{EI}(0.6) = (0.82 - 0.8) \times \Phi\left(\frac{0.02}{\sqrt{0.05}}\right) + \sqrt{0.05} \times \phi\left(\frac{0.02}{\sqrt{0.05}}\right)$$
$$\approx 0.02 \times 0.54 + 0.22 \times 0.39 \approx 0.097$$

建议在 $x = 0.6$ 附近继续探索。

</details>

**练习 18.7**：分析一个自动驾驶场景中的动态 shape 问题。检测到的目标数量在 [0, 50] 之间变化，每帧处理时间限制为 33ms。设计一个编译策略，确保 99% 的帧满足实时性要求。已知：
- 目标数 ≤ 10 的概率：80%
- 目标数 11-30 的概率：15%
- 目标数 31-50 的概率：5%

*Hint*：考虑分层编译和降级策略。

<details>
<summary>答案</summary>

分层策略设计：

1. **L0 层（快速路径）**：
   - 针对 0-10 个目标深度优化
   - 执行时间：< 20ms
   - 覆盖 80% 情况

2. **L1 层（标准路径）**：
   - 针对 11-30 个目标优化
   - 执行时间：< 30ms
   - 覆盖 15% 情况

3. **L2 层（降级路径）**：
   - 31-50 个目标
   - 使用低精度/跳帧策略
   - 确保 < 33ms

时间预算分配：
- 80% × 20ms + 15% × 30ms + 5% × 33ms = 22.15ms 平均延迟

99 分位延迟 ≤ 33ms，满足要求。

降级策略：
- 当目标数 > 40 时，降低检测精度
- 当连续 2 帧超时，跳过非关键目标

</details>

## 常见陷阱与错误

1. **过度特化陷阱**：为每个轻微的形状变化都生成特化代码，导致编译开销爆炸和缓存污染。应该设置合理的相似度阈值。

2. **缓存一致性问题**：在分布式环境中，不同节点的缓存可能不一致。必须实现严格的版本控制和同步机制。

3. **内存泄漏**：动态编译的代码如果没有正确释放，会导致内存持续增长。需要实现代码生命周期管理。

4. **性能抖动**：频繁的重编译会导致性能不稳定。应该实现编译决策的滞后机制。

5. **预测模型过拟合**：在线学习的预测模型可能过度适应特定工作负载。需要定期重置或使用正则化。

6. **死锁风险**：多线程环境下的缓存锁可能导致死锁。建议使用无锁数据结构或细粒度锁。

7. **形状爆炸**：某些模型的形状空间极大（如 NLP 的变长输入）。必须实现有效的桶化和泛化策略。

## 最佳实践检查清单

### 设计阶段
- [ ] 定义清晰的形状分类标准和相似度度量
- [ ] 设计多级缓存架构，平衡命中率和内存占用
- [ ] 建立性能模型，包括编译成本和执行收益
- [ ] 规划降级策略，处理极端情况

### 实现阶段
- [ ] 实现高效的形状哈希和查找机制
- [ ] 使用无锁或细粒度锁减少并发开销
- [ ] 实现增量编译，复用已有编译结果
- [ ] 添加性能计数器，跟踪关键指标

### 优化阶段
- [ ] 分析形状分布，优化桶化策略
- [ ] 调优重编译阈值，平衡性能和开销
- [ ] 实现预测性编译，提前准备热点形状
- [ ] 优化内存布局，减少缓存失效

### 监控阶段
- [ ] 监控缓存命中率和编译频率
- [ ] 跟踪 P50/P90/P99 延迟指标
- [ ] 检测内存泄漏和资源耗尽
- [ ] 分析性能瓶颈，持续优化

### 测试阶段
- [ ] 测试各种形状组合和变化模式
- [ ] 验证缓存一致性和正确性
- [ ] 压力测试，确保系统稳定性
- [ ] 基准测试，对比静态编译性能