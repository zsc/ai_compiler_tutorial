# 第 12 章：JIT 编译技术

## 章节大纲

### 12.1 JIT vs AOT 权衡
- JIT 编译的基本原理
- AOT 编译的优势与局限
- 混合编译策略
- 性能与部署权衡

### 12.2 热点检测与优化
- 分析型热点检测
- 采样型热点检测
- 热度阈值设计
- 自适应优化策略

### 12.3 编译缓存管理
- 缓存键设计
- 持久化策略
- 内存管理
- 版本兼容性

### 12.4 分层编译策略
- 编译层级设计
- 晋升与降级机制
- 优化级别选择
- 资源分配策略

---

## 开篇

即时编译（JIT）技术是 AI 编译器中平衡编译开销与运行性能的关键技术。不同于传统编译器的静态优化，JIT 编译器能够利用运行时信息进行动态优化，这对于处理动态 shape、稀疏模式和自适应计算尤为重要。本章将深入探讨 JIT 编译在 AI 场景中的设计与实现，特别关注自动驾驶和具身智能等对延迟敏感的应用场景。

### 学习目标
1. 理解 JIT 与 AOT 编译的本质区别和适用场景
2. 掌握热点检测的数学模型和工程实现
3. 设计高效的编译缓存系统
4. 实现多层级编译策略以平衡性能与开销

## 12.1 JIT vs AOT 权衡

### 12.1.1 基本概念对比

提前编译（Ahead-of-Time, AOT）和即时编译（Just-In-Time, JIT）代表了两种不同的编译哲学：

**AOT 编译特征：**
- 编译发生在部署前
- 可进行耗时的全局优化
- 生成的代码大小可预测
- 启动延迟低
- 无法利用运行时信息

**JIT 编译特征：**
- 编译发生在运行时
- 可利用实际运行数据优化
- 动态适应执行模式
- 启动延迟高（冷启动问题）
- 内存开销包含编译器本身

### 12.1.2 性能模型

设运行时间为 $T_{total}$，可以建模为：

$$T_{total} = T_{compile} + T_{execute}$$

对于 AOT：
$$T_{AOT} = 0 + N \cdot t_{AOT}$$

对于 JIT：
$$T_{JIT} = \sum_{i=1}^{k} c_i + N \cdot t_{JIT}$$

其中：
- $N$ 是执行次数
- $t_{AOT}$ 是 AOT 编译代码的单次执行时间
- $t_{JIT}$ 是 JIT 优化后代码的执行时间
- $c_i$ 是第 $i$ 次编译的开销
- $k$ 是总编译次数

当 $t_{JIT} < t_{AOT}$ 且 $N$ 足够大时，JIT 的优势显现：
$$N > \frac{\sum_{i=1}^{k} c_i}{t_{AOT} - t_{JIT}}$$

### 12.1.3 混合编译策略

现代 AI 编译器通常采用混合策略：

```
编译决策树：
         输入程序
            |
      静态形状分析
         /     \
    固定形状   动态形状
       |         |
   AOT编译    JIT框架
               |
          运行时特化
```

**分层决策模型：**

定义收益函数 $B(op, mode)$：
$$B(op, mode) = P_{hot}(op) \cdot G_{opt}(op, mode) - C_{compile}(op, mode)$$

其中：
- $P_{hot}(op)$ 是算子成为热点的概率
- $G_{opt}(op, mode)$ 是优化后的性能增益
- $C_{compile}(op, mode)$ 是编译成本

### 12.1.4 AI 场景的特殊考虑

**自动驾驶场景：**
- 实时性要求严格（< 100ms 端到端延迟）
- 倾向 AOT 编译关键路径
- JIT 用于非关键的后处理

**具身智能场景：**
- 任务多样性高
- 动态 shape 频繁
- JIT 编译更有优势

**延迟敏感度分析：**

设系统延迟预算为 $L_{budget}$，第 $i$ 个算子的延迟为 $l_i$：

$$\sum_{i \in critical\_path} l_i \leq L_{budget}$$

JIT 编译引入的额外延迟 $l_{JIT}$ 必须满足：
$$l_{JIT} < L_{budget} - \sum_{i \in critical\_path} l_i^{min}$$

## 12.2 热点检测与优化

### 12.2.1 热点检测机制

热点（Hotspot）是程序中频繁执行的代码区域。准确识别热点是 JIT 优化的前提。

**计数器方法：**

每个算子维护执行计数器 $count(op)$：
$$hotness(op) = \frac{count(op)}{\sum_{i} count(op_i)}$$

当 $hotness(op) > \theta_{hot}$ 时触发编译。

**采样方法：**

使用概率采样减少开销：
$$P_{sample} = \min(1, \frac{k}{f_{exec}})$$

其中 $f_{exec}$ 是执行频率，$k$ 是采样常数。

### 12.2.2 热度传播模型

考虑计算图 $\mathcal{G} = (V, E)$，热度在图中传播：

$$H_v^{(t+1)} = \alpha \cdot H_v^{(t)} + (1-\alpha) \cdot \sum_{u \in pred(v)} \frac{H_u^{(t)}}{|succ(u)|}$$

其中：
- $H_v^{(t)}$ 是节点 $v$ 在时刻 $t$ 的热度
- $\alpha$ 是衰减因子（通常取 0.9）
- $pred(v)$ 和 $succ(u)$ 分别是前驱和后继节点集

### 12.2.3 自适应阈值调整

静态阈值可能导致过早或过晚编译。自适应阈值根据历史数据动态调整：

$$\theta_{hot}^{(t+1)} = \theta_{hot}^{(t)} + \eta \cdot (R_{actual} - R_{target})$$

其中：
- $R_{actual}$ 是实际编译收益
- $R_{target}$ 是目标收益
- $\eta$ 是学习率

**收益度量：**
$$R = \frac{t_{before} - t_{after}}{t_{compile}}$$

### 12.2.4 多维度热点分析

除执行频率外，还需考虑：

1. **时间占比：** $T_{ratio} = \frac{t_{op}}{\sum_i t_i}$
2. **内存带宽：** $B_{ratio} = \frac{bytes_{op}}{bandwidth_{peak}}$
3. **计算密度：** $C_{density} = \frac{FLOPs}{bytes}$

综合热度评分：
$$Score = w_1 \cdot count + w_2 \cdot T_{ratio} + w_3 \cdot B_{ratio} + w_4 \cdot C_{density}$$

权重 $w_i$ 根据硬件特性调整。

## 12.3 编译缓存管理

### 12.3.1 缓存键设计

缓存键必须唯一标识编译结果：

$$Key = Hash(IR, Shape, OptLevel, Target)$$

**多级缓存键：**
- L1：精确匹配 - $(IR_{hash}, shape_{exact}, opt_{level})$
- L2：形状类匹配 - $(IR_{hash}, shape_{bucket}, opt_{level})$
- L3：结构匹配 - $(IR_{structure}, *, opt_{level})$

### 12.3.2 缓存替换策略

**LRU-K 算法：**

记录最近 $K$ 次访问时间，按第 $K$ 次访问时间排序：
$$Priority(item) = \begin{cases}
    t_K & \text{if } count \geq K \\
    -\infty & \text{otherwise}
\end{cases}$$

**成本感知替换：**

考虑编译成本的替换策略：
$$Value(item) = \frac{hit\_count \cdot speedup}{size + compile\_cost}$$

替换 $Value$ 最小的项。

### 12.3.3 持久化与共享

**分层存储：**
```
    内存缓存 (L1)
         |
    本地磁盘 (L2)
         |
   分布式缓存 (L3)
```

**版本兼容性：**

缓存项包含版本信息：
$$CacheEntry = \{Key, Binary, Version, Metadata\}$$

版本检查：
$$Compatible(v_1, v_2) = major(v_1) = major(v_2) \land minor(v_1) \leq minor(v_2)$$

### 12.3.4 内存管理

**内存预算分配：**

设总内存预算为 $M_{total}$，分配给缓存的比例为 $\rho$：
$$M_{cache} = \rho \cdot M_{total}$$

动态调整 $\rho$：
$$\rho^{(t+1)} = \rho^{(t)} + \beta \cdot (hit\_rate - target\_rate)$$

**碎片整理：**

当碎片率超过阈值时触发整理：
$$Fragmentation = 1 - \frac{\sum_i size_i}{M_{allocated}}$$

## 12.4 分层编译策略

### 12.4.1 编译层级设计

典型的分层编译包含以下层级：

**Layer 0 - 解释执行：**
- 零编译开销
- 最慢执行速度
- 收集 profiling 信息

**Layer 1 - 基础编译：**
- 快速编译（< 10ms）
- 基本优化（常量折叠、死代码消除）
- 生成基础机器码

**Layer 2 - 优化编译：**
- 中等编译时间（10-100ms）
- 循环优化、向量化
- 算子融合

**Layer 3 - 激进优化：**
- 长编译时间（> 100ms）
- 全局优化、自动调优
- 特化代码生成

### 12.4.2 晋升与降级机制

**晋升条件：**

从层级 $i$ 晋升到 $i+1$ 的条件：
$$Promote(op, i \to i+1) = \begin{cases}
    true & \text{if } count(op) > \theta_i \land benefit(op) > B_{min} \\
    false & \text{otherwise}
\end{cases}$$

晋升阈值呈指数增长：
$$\theta_i = \theta_0 \cdot \gamma^i$$

其中 $\gamma > 1$（通常取 10）。

**降级触发：**

当检测到执行模式变化时降级：
$$Demote(op) = \|profile_{current} - profile_{compiled}\| > \epsilon$$

Profile 向量包含：
- 输入形状分布
- 分支概率
- 内存访问模式

### 12.4.3 优化级别选择

**成本-收益分析：**

选择优化级别 $l^*$ 使得净收益最大：
$$l^* = \arg\max_l \left[ N_{expected} \cdot (t_0 - t_l) - C_l \right]$$

其中：
- $N_{expected}$ 是预期执行次数
- $t_l$ 是级别 $l$ 的执行时间
- $C_l$ 是级别 $l$ 的编译成本

**执行次数预测：**

使用指数加权移动平均：
$$N_{expected}^{(t+1)} = \alpha \cdot N_{actual}^{(t)} + (1-\alpha) \cdot N_{expected}^{(t)}$$

### 12.4.4 资源分配策略

**编译线程池管理：**

设系统有 $P$ 个处理器，分配策略：
- 执行线程：$P_{exec} = \lceil 0.8P \rceil$
- 编译线程：$P_{compile} = \lfloor 0.2P \rfloor$

动态调整基于队列长度：
$$P_{compile}^{(t+1)} = \min(P/2, P_{compile}^{(t)} + sign(Q_{length} - Q_{target}))$$

**内存预算分配：**

各层级内存分配比例：
$$M_i = M_{total} \cdot \frac{w_i \cdot usage_i}{\sum_j w_j \cdot usage_j}$$

权重 $w_i$ 反映层级重要性：
- Layer 0: $w_0 = 0.1$
- Layer 1: $w_1 = 0.3$
- Layer 2: $w_2 = 0.4$
- Layer 3: $w_3 = 0.2$

## 12.5 JIT 在 AI 场景的特殊优化

### 12.5.1 动态 Shape 特化

**Shape 桶化策略：**

将连续的 shape 空间离散化为桶：
$$Bucket(s) = \left\lfloor \frac{\log_2(s)}{\delta} \right\rfloor \cdot \delta$$

其中 $\delta$ 控制桶的粒度。

**特化决策：**

当某个 shape 的频率超过阈值时特化：
$$Specialize(shape) = \frac{count(shape)}{count_{total}} > \theta_{spec}$$

### 12.5.2 算子融合的 JIT 优化

**动态融合模式识别：**

运行时识别可融合的算子序列：
$$Fusible(op_1, op_2) = Compatible(output_1, input_2) \land NoAlias(op_1, op_2)$$

融合收益评估：
$$Benefit_{fusion} = BW_{saved} - Overhead_{fusion}$$

其中：
$$BW_{saved} = size(intermediate) \cdot (read + write)$$

### 12.5.3 投机编译

**预测性编译：**

基于历史模式预测未来需要的编译：
$$P(shape_{next} = s | history) = \frac{count(pattern \to s)}{\sum_i count(pattern \to s_i)}$$

当 $P(s) > \theta_{spec}$ 时触发投机编译。

**资源约束下的投机：**

限制投机编译的资源使用：
$$\sum_{s \in speculative} C(s) \leq \beta \cdot C_{available}$$

其中 $\beta \in [0, 1]$ 是投机编译的资源比例上限。

## 12.6 性能分析与调优

### 12.6.1 JIT 开销分解

总开销可分解为：
$$Overhead_{JIT} = T_{detect} + T_{compile} + T_{install} + T_{deopt}$$

各部分典型占比：
- 热点检测：5-10%
- 编译：70-80%
- 代码安装：5-10%
- 去优化：5-15%

### 12.6.2 编译时间预测

使用线性模型预测编译时间：
$$T_{compile} = \alpha \cdot |IR| + \beta \cdot OptLevel + \gamma \cdot Complexity + \epsilon$$

其中 $Complexity$ 可用循环嵌套深度、数据依赖复杂度等度量。

### 12.6.3 性能监控指标

关键监控指标：
1. **编译吞吐量：** $Throughput = \frac{BytesCompiled}{Time}$
2. **缓存命中率：** $HitRate = \frac{Hits}{Hits + Misses}$
3. **晋升率：** $PromotionRate = \frac{Promotions}{Executions}$
4. **去优化率：** $DeoptRate = \frac{Deopts}{OptimizedExecutions}$

## 本章小结

JIT 编译技术在 AI 编译器中扮演着至关重要的角色，特别是在处理动态 shape 和自适应优化场景。本章核心要点：

1. **JIT vs AOT 权衡：** JIT 适合动态场景但有冷启动开销，AOT 适合静态场景和实时系统
2. **热点检测：** 准确识别热点是 JIT 优化的基础，需要平衡检测开销和准确性
3. **缓存管理：** 高效的缓存系统可以显著减少重复编译开销
4. **分层编译：** 通过多级优化平衡编译开销和执行性能
5. **AI 特殊优化：** 动态 shape 特化、算子融合和投机编译是 AI 场景的关键技术

关键公式回顾：
- 性能平衡点：$N > \frac{\sum_{i=1}^{k} c_i}{t_{AOT} - t_{JIT}}$
- 热度传播：$H_v^{(t+1)} = \alpha \cdot H_v^{(t)} + (1-\alpha) \cdot \sum_{u} \frac{H_u^{(t)}}{|succ(u)|}$
- 优化级别选择：$l^* = \arg\max_l \left[ N_{expected} \cdot (t_0 - t_l) - C_l \right]$

## 练习题

### 基础题

**练习 12.1：** 某 AI 模型的推理包含 100 个算子，其中 20 个算子占总执行时间的 80%。如果 JIT 编译每个算子需要 50ms，优化后性能提升 2 倍，每个算子平均执行 1000 次，计算 JIT 相比解释执行的收益。

*Hint：分别计算关键算子和非关键算子的优化收益。*

<details>
<summary>参考答案</summary>

设总执行时间为 $T$，则：
- 20 个关键算子：执行时间 $0.8T$，单次 $\frac{0.8T}{20 \times 1000} = \frac{T}{25000}$
- 80 个非关键算子：执行时间 $0.2T$，单次 $\frac{0.2T}{80 \times 1000} = \frac{T}{400000}$

JIT 后：
- 关键算子：编译 $20 \times 50ms = 1s$，执行 $0.8T/2 = 0.4T$
- 总时间：$1s + 0.4T + 0.2T = 1s + 0.6T$

收益：当 $T > 2.5s$ 时，JIT 有正收益。
实际收益率：$(T - 1 - 0.6T)/T = 0.4 - 1/T$

</details>

**练习 12.2：** 设计一个简单的热点检测算法，使用计数器方法，当算子执行次数超过 $\sqrt{N}$（$N$ 为总执行次数）时触发编译。证明这个阈值的合理性。

*Hint：考虑帕累托分布（80/20 法则）。*

<details>
<summary>参考答案</summary>

假设执行次数服从帕累托分布：$P(X > x) = (x_m/x)^\alpha$

对于 80/20 法则，$\alpha \approx 1.16$。

设有 $M$ 个算子，总执行 $N$ 次。热点算子数量约为 $0.2M$，执行 $0.8N$ 次。

平均每个热点算子执行：$\frac{0.8N}{0.2M} = \frac{4N}{M}$

选择阈值 $\theta = \sqrt{N}$，当 $M < 16\sqrt{N}$ 时，所有热点都会被检测到。

这在实践中通常成立，因为算子数量相对执行次数的平方根较小。

</details>

**练习 12.3：** 某编译缓存使用 LRU 策略，缓存大小为 100MB，平均每个编译结果 5MB。如果访问模式符合 Zipf 分布（$P(i) \propto 1/i$），计算前 20 个最热项的缓存命中率。

*Hint：计算 Zipf 分布的累积概率。*

<details>
<summary>参考答案</summary>

缓存可存储：$100MB / 5MB = 20$ 个项。

Zipf 分布：$P(i) = \frac{1/i}{\sum_{j=1}^{n} 1/j} = \frac{1/i}{H_n}$

其中 $H_n$ 是调和数。

前 20 项的命中率：
$$HitRate = \frac{\sum_{i=1}^{20} 1/i}{H_n}$$

对于大 $n$，$H_n \approx \ln(n) + \gamma$（$\gamma \approx 0.577$）

若总共 1000 个不同编译结果：
$$HitRate = \frac{H_{20}}{H_{1000}} \approx \frac{3.598}{7.486} \approx 48\%$$

</details>

### 挑战题

**练习 12.4：** 设计一个自适应的分层编译策略，根据算子的执行频率和计算复杂度动态选择编译层级。考虑 4 个层级，编译时间分别为 1ms、10ms、100ms、1000ms，性能提升分别为 1.2x、1.5x、2x、3x。

*Hint：建立成本模型，使用动态规划求解最优策略。*

<details>
<summary>参考答案</summary>

定义状态：$V(n, l)$ = 执行 $n$ 次、当前层级 $l$ 的最小总时间。

状态转移：
$$V(n, l) = \min_{l' \geq l} \left\{ C_{l'} + n \cdot T_{l'} + V(n - n_{current}, l') \right\}$$

其中：
- $C_{l'}$ 是编译到层级 $l'$ 的成本
- $T_{l'}$ 是层级 $l'$ 的单次执行时间
- $n_{current}$ 是当前批次执行次数

动态策略：
1. 初始使用 Layer 0（解释执行）
2. 当 $n > 10$ 时，评估是否编译到 Layer 1
3. 当 $n > 100$ 时，评估是否编译到 Layer 2
4. 当 $n > 1000$ 时，评估是否编译到 Layer 3

决策函数：
$$Compile(n, l_{current}, l_{target}) = n \cdot (T_{current} - T_{target}) > C_{target}$$

</details>

**练习 12.5：** 分析投机编译的风险与收益。假设有 $K$ 种可能的 shape，每种概率为 $p_i$，编译成本为 $C$，优化收益为 $B$。设计最优的投机编译策略。

*Hint：将问题建模为背包问题的变体。*

<details>
<summary>参考答案</summary>

期望收益：
$$E[Benefit] = \sum_{i=1}^{K} p_i \cdot (B_i \cdot I_{compiled}(i) - C \cdot I_{compile}(i))$$

其中 $I_{compiled}(i)$ 表示 shape $i$ 是否已编译。

约束条件（资源限制）：
$$\sum_{i=1}^{K} I_{compile}(i) \cdot C \leq Budget$$

这是一个 0-1 背包问题。最优策略：
1. 计算收益密度：$\rho_i = \frac{p_i \cdot B_i}{C}$
2. 按 $\rho_i$ 降序排序
3. 贪心选择直到资源耗尽

当 shape 分布不确定时，使用置信区间：
$$p_i \in [\hat{p}_i - \epsilon, \hat{p}_i + \epsilon]$$

采用鲁棒优化：
$$\max_{\{I_i\}} \min_{p \in \mathcal{P}} E[Benefit]$$

</details>

**练习 12.6：** 在自动驾驶场景中，感知模块要求 99.9% 的推理在 50ms 内完成。设计一个 JIT 策略，保证实时性的同时最大化性能。考虑编译时间的尾部延迟。

*Hint：使用分位数优化和降级机制。*

<details>
<summary>参考答案</summary>

设计两级系统：
1. **快速路径：** AOT 编译的基础版本，保证 $t_{AOT} < 45ms$
2. **优化路径：** JIT 编译的优化版本，目标 $t_{JIT} < 30ms$

策略：
1. 所有请求先走快速路径
2. 后台异步 JIT 编译
3. 编译完成后切换到优化路径

尾部延迟控制：
- 设置编译超时：$T_{timeout} = 5ms$
- 使用优先级队列，关键算子优先
- 监控 P99.9 延迟：
  $$P_{99.9}(latency) = \max(t_{AOT}, p_{switch} \cdot t_{switch} + t_{JIT})$$

其中 $p_{switch}$ 是切换期间的请求比例。

降级机制：
- 当检测到延迟尖峰时，立即降级到 AOT 版本
- 降级条件：$latency_{current} > 0.9 \cdot budget$

</details>

## 常见陷阱与错误 (Gotchas)

1. **过早优化陷阱**
   - 错误：设置过低的编译阈值
   - 后果：编译开销超过性能收益
   - 解决：使用自适应阈值，基于历史数据调整

2. **缓存键设计错误**
   - 错误：缓存键未包含所有影响因素
   - 后果：使用错误的编译结果，导致计算错误
   - 解决：完整的键设计，包括 IR、shape、优化级别、硬件目标

3. **内存泄漏**
   - 错误：编译结果未及时释放
   - 后果：内存持续增长，最终 OOM
   - 解决：实现严格的生命周期管理，使用引用计数

4. **热点检测偏差**
   - 错误：只考虑执行次数，忽略执行时间
   - 后果：优化了错误的目标
   - 解决：综合考虑频率、时间占比、内存带宽等因素

5. **编译风暴**
   - 错误：大量算子同时触发编译
   - 后果：系统卡顿，响应时间激增
   - 解决：限制并发编译数，使用编译队列

6. **版本不兼容**
   - 错误：使用旧版本编译的缓存
   - 后果：运行时错误或性能退化
   - 解决：严格的版本检查，自动缓存失效机制

## 最佳实践检查清单

### 设计阶段
- [ ] 明确 JIT vs AOT 的选择标准
- [ ] 设计多级编译层次
- [ ] 规划缓存策略和容量
- [ ] 定义热点检测指标
- [ ] 设计降级和回退机制

### 实现阶段
- [ ] 实现准确的 profiling 机制
- [ ] 优化编译器自身的性能
- [ ] 实现高效的缓存系统
- [ ] 添加编译任务调度器
- [ ] 实现版本兼容性检查

### 优化阶段
- [ ] 分析编译开销分布
- [ ] 优化热点检测算法
- [ ] 调优缓存替换策略
- [ ] 实现投机编译
- [ ] 优化层级晋升策略

### 监控阶段
- [ ] 监控编译吞吐量
- [ ] 跟踪缓存命中率
- [ ] 分析去优化频率
- [ ] 监控内存使用
- [ ] 跟踪 P99 延迟

### 调试阶段
- [ ] 记录编译决策日志
- [ ] 保存 profiling 数据
- [ ] 支持强制编译模式
- [ ] 提供缓存统计信息
- [ ] 实现性能回归检测