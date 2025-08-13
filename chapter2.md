# 第 2 章：中间表示（IR）设计

## 章节大纲

### 2.1 开篇：IR 在 AI 编译器中的核心地位
- IR 设计决定编译器能力边界
- 自动驾驶与具身智能场景的 IR 需求

### 2.2 多层 IR 的必要性
- 2.2.1 抽象层次与优化机会
- 2.2.2 渐进式降级（Progressive Lowering）
- 2.2.3 层间转换的正确性保证
- 2.2.4 200T 模型的 IR 层次设计

### 2.3 图 IR vs 指令 IR
- 2.3.1 数据流图表示的优势与局限
- 2.3.2 控制流的表达方式对比
- 2.3.3 混合表示：最佳实践
- 2.3.4 大规模并行场景的选择策略

### 2.4 SSA 形式在 AI 编译器中的应用
- 2.4.1 传统 SSA 与张量 SSA 的区别
- 2.4.2 φ 节点在动态控制流中的处理
- 2.4.3 内存 SSA 与别名分析
- 2.4.4 梯度计算中的 SSA 转换

### 2.5 MLIR 方言设计原则
- 2.5.1 方言的层次结构设计
- 2.5.2 类型系统与属性设计
- 2.5.3 Operation 语义定义
- 2.5.4 方言间的转换模式
- 2.5.5 自定义方言的设计决策

### 2.6 本章小结

### 2.7 练习题

### 2.8 常见陷阱与错误

### 2.9 最佳实践检查清单

---

## 2.1 开篇：IR 在 AI 编译器中的核心地位

中间表示（Intermediate Representation，IR）是 AI 编译器的灵魂。它不仅是前端语言与后端硬件之间的桥梁，更是各种优化变换的操作对象。对于自动驾驶和具身智能等实时 AI 系统，IR 设计的优劣直接决定了模型部署的性能上限、内存效率和功耗表现。本章将深入探讨 AI 编译器 IR 设计的核心原则，特别关注如何通过精心设计的多层 IR 体系支撑 200T 参数级模型的高效编译。

在传统编译器中，IR 主要关注标量和简单数据结构的操作。而 AI 编译器的 IR 必须原生支持高维张量运算、复杂的并行模式以及异构硬件的特性。以自动驾驶场景为例，感知模块的卷积运算、决策模块的 Transformer 推理、规划模块的图神经网络，每种计算模式对 IR 的要求都不相同。一个优秀的 IR 设计必须在表达能力、优化机会和编译效率之间找到平衡点。

## 2.2 多层 IR 的必要性

### 2.2.1 抽象层次与优化机会

多层 IR 架构是现代 AI 编译器的标准设计范式。不同抽象层次的 IR 承载着不同的优化责任：

**高层 IR（Graph Level）**：保留完整的语义信息，适合进行算法级优化。在这一层，编译器可以识别：
- 算子融合机会：相邻的 element-wise 操作可以合并
- 常量折叠：编译时可计算的子图
- 代数简化：利用数学恒等式简化计算
- 死代码消除：移除不影响输出的计算

**中层 IR（Loop Level）**：暴露循环结构和内存访问模式，适合进行循环优化和内存优化：
- 循环分块（tiling）：改善缓存局部性
- 循环融合与分裂：平衡并行性和局部性
- 向量化：利用 SIMD 指令
- 预取优化：隐藏内存延迟

**低层 IR（Instruction Level）**：接近目标硬件，进行特定硬件优化：
- 指令选择：选择最优的硬件指令
- 寄存器分配：最小化内存访问
- 指令调度：利用指令级并行
- 硬件特性利用：如 Tensor Core、Matrix Engine

### 2.2.2 渐进式降级（Progressive Lowering）

渐进式降级是多层 IR 系统的核心机制。每次降级都是一次信息的精化和特化过程：

$$\mathcal{IR}_{high} \xrightarrow{\phi_1} \mathcal{IR}_{mid} \xrightarrow{\phi_2} \mathcal{IR}_{low} \xrightarrow{\phi_3} \text{Machine Code}$$

其中每个转换函数 $\phi_i$ 必须保证语义等价性：

$$\forall p \in \mathcal{IR}_i, \text{Semantics}(p) = \text{Semantics}(\phi_i(p))$$

降级过程中的关键决策点包括：

1. **内存布局确定**：从逻辑张量到物理内存布局
2. **并行策略选择**：从抽象并行到具体的线程/块映射
3. **数值精度转换**：从高精度到混合精度
4. **控制流具体化**：从高级控制流到跳转指令

### 2.2.3 层间转换的正确性保证

确保 IR 转换的正确性是编译器可靠性的基础。主要验证方法包括：

**形式化验证**：通过定理证明器验证转换规则的正确性。设转换前后的程序分别为 $P$ 和 $P'$，需要证明：

$$\forall \text{input} \in \mathcal{D}, \llbracket P \rrbracket(\text{input}) = \llbracket P' \rrbracket(\text{input})$$

**差分测试**：通过对比不同优化级别的输出验证正确性。容许数值误差 $\epsilon$：

$$\|output_{opt} - output_{ref}\|_{\infty} < \epsilon$$

**不变量检查**：在转换前后检查关键属性保持不变，如：
- 张量形状一致性
- 数据依赖关系保持
- 内存访问合法性
- 数值范围约束

### 2.2.4 200T 模型的 IR 层次设计

超大规模模型对 IR 设计提出了特殊挑战：

**分片表示**：在高层 IR 中需要表示跨节点的张量分片：

```
Tensor<shape=[B, S, H], sharding=[DP:8, TP:4, PP:16]>
```

其中 DP、TP、PP 分别表示数据并行、张量并行和流水线并行的分片维度。

**通信原语**：IR 必须原生支持集合通信操作：
- AllReduce：梯度聚合
- AllGather：激活值收集
- ReduceScatter：分布式归约

**内存层级感知**：IR 需要区分不同的内存层级：
- HBM（高带宽内存）
- L2 Cache
- Shared Memory / Local Memory
- Register File

对于 200T 模型，典型的 IR 层次设计如下：

```
Level 0: Model Definition IR (PyTorch/JAX)
         ↓
Level 1: Distributed Graph IR (分片策略已确定)
         ↓
Level 2: Kernel Fusion IR (算子已融合)
         ↓
Level 3: Memory Planning IR (内存已规划)
         ↓
Level 4: Hardware-specific IR (CUDA/XLA HLO)
         ↓
Level 5: Assembly/Binary
```

每层的内存占用估算模型：

$$M_{total} = M_{params} + M_{activations} + M_{gradients} + M_{optimizer} + M_{buffer}$$

其中对于 200T 模型：
- $M_{params} \approx 400TB$ (FP16)
- $M_{activations} \approx O(B \times L \times H)$，B 为批大小，L 为序列长度，H 为隐藏维度
- $M_{gradients} = M_{params}$
- $M_{optimizer} \geq 2 \times M_{params}$ (Adam)

## 2.3 图 IR vs 指令 IR

### 2.3.1 数据流图表示的优势与局限

**图 IR 的核心优势**：

图 IR 将计算表示为有向无环图（DAG）或更一般的数据流图，节点代表操作，边代表数据依赖。这种表示特别适合 AI 工作负载：

1. **并行性显式化**：没有数据依赖的节点可以并行执行
2. **优化机会易识别**：模式匹配算法可以高效识别子图模式
3. **硬件映射灵活**：易于进行设备分配和算子调度
4. **可视化友好**：便于调试和性能分析

图 IR 的数学表示：
$$\mathcal{G} = (V, E, \psi, \tau)$$

其中：
- $V$ 是操作节点集合
- $E \subseteq V \times V$ 是数据流边
- $\psi: V \rightarrow \mathcal{O}$ 将节点映射到操作类型
- $\tau: E \rightarrow \mathcal{T}$ 将边映射到张量类型

**图 IR 的局限性**：

1. **控制流表达困难**：动态控制流需要特殊节点（如 Switch、Merge）
2. **内存管理不直观**：难以表达精确的内存分配和释放时机
3. **副作用难处理**：I/O 操作、随机数生成等需要额外的依赖边
4. **调度约束表达**：难以表达"必须在 X 之后 Y 之前执行"等约束

### 2.3.2 控制流的表达方式对比

**图 IR 中的控制流**：

使用控制依赖边和特殊节点：
```
     Cond
    /    \
   T      F
   |      |
Switch  Switch
   |      |
 Body1  Body2
   \    /
    Merge
```

条件执行的语义：
$$\text{If}(c, t, f) = \begin{cases} t() & \text{if } c = \text{true} \\ f() & \text{if } c = \text{false} \end{cases}$$

**指令 IR 中的控制流**：

使用基本块和跳转指令：
```
BB0:
  %cond = compare %x, %y
  br %cond, BB1, BB2

BB1:  ; true branch
  %result1 = ...
  br BB3

BB2:  ; false branch
  %result2 = ...
  br BB3

BB3:  ; merge
  %result = phi [%result1, BB1], [%result2, BB2]
```

### 2.3.3 混合表示：最佳实践

现代 AI 编译器通常采用混合策略：

**分层混合**：
- 高层使用图 IR（如 TensorFlow Graph、PyTorch FX）
- 低层使用指令 IR（如 LLVM IR、SPIR-V）

**区域混合**：
- 数据并行区域用图 IR
- 控制密集区域用指令 IR

**按需转换**：
```
Graph Region {
  %a = Conv2D(%input, %weight)
  %b = ReLU(%a)
  
  CFG Region {
    for i in range(N):
      %c[i] = MatMul(%b, %w[i])
  }
  
  %output = Concat(%c)
}
```

### 2.3.4 大规模并行场景的选择策略

对于自动驾驶和具身智能的不同模块，IR 选择策略不同：

**感知模块（CNN 为主）**：
- 推荐：图 IR
- 原因：规则的数据流，少量控制流
- 优化重点：算子融合、内存复用

**决策模块（Transformer 为主）**：
- 推荐：混合 IR
- 原因：注意力机制有动态性，但主体仍是矩阵运算
- 优化重点：FlashAttention 类优化、KV Cache 管理

**规划模块（搜索算法）**：
- 推荐：指令 IR
- 原因：复杂的控制流和递归结构
- 优化重点：分支预测、投机执行

性能模型对比：

设批处理大小为 $B$，模型深度为 $L$，隐藏维度为 $H$：

图 IR 调度开销：$O(|V| + |E|)$
指令 IR 调度开销：$O(|BB| \times |I|)$

其中 $|BB|$ 是基本块数量，$|I|$ 是平均指令数。

对于典型的 Transformer 模型：
- 图 IR：$|V| = O(L)$，$|E| = O(L)$
- 指令 IR：$|BB| = O(L)$，$|I| = O(H^2)$

因此对于大模型，图 IR 在编译时间上有优势。

## 2.4 SSA 形式在 AI 编译器中的应用

### 2.4.1 传统 SSA 与张量 SSA 的区别

静态单赋值（Static Single Assignment，SSA）形式是编译器优化的基石。在传统编译器中，SSA 确保每个变量只被赋值一次，极大简化了数据流分析。但在 AI 编译器中，我们处理的是张量而非标量，这带来了新的挑战和机遇。

**传统 SSA**：
```
x = 1           =>    x_0 = 1
y = x + 2             y_0 = x_0 + 2
x = y * 3             x_1 = y_0 * 3
z = x + y             z_0 = x_1 + y_0
```

**张量 SSA 的特殊性**：

1. **部分更新问题**：张量经常只更新部分元素
   $$T'[i:j, k:l] = f(T[i:j, k:l])$$
   
   SSA 表示需要引入特殊的更新操作：
   $$T_1 = \text{update}(T_0, \text{slice}=(i:j, k:l), \text{value}=f(T_0[i:j, k:l]))$$

2. **内存别名复杂性**：张量的视图（view）和切片（slice）创建别名关系
   ```
   A_0 = allocate([1000, 1000])
   B_0 = view(A_0, [100, 10000])  # B和A共享内存
   B_1 = B_0 + 1                   # 影响A的内容
   ```

3. **生命周期管理**：大张量的生命周期直接影响内存使用
   - 需要精确的 def-use 链分析
   - 支持原地操作（in-place operation）优化

### 2.4.2 φ 节点在动态控制流中的处理

φ（phi）节点用于在控制流汇合点选择正确的值。在 AI 编译器中，φ 节点需要处理张量级别的选择：

**标量 φ 节点**：
```
BB_merge:
  x_2 = φ(x_0 from BB_then, x_1 from BB_else)
```

**张量 φ 节点的挑战**：

1. **内存开销**：简单复制大张量代价高昂
2. **条件写入**：需要支持条件选择写入
   $$T_{out} = \text{where}(condition, T_{true}, T_{false})$$

3. **优化机会**：
   - 如果分支中张量大部分相同，可以只更新差异部分
   - 可以延迟 φ 节点的实际执行直到真正使用

**动态形状下的 φ 节点**：

当张量形状在运行时才能确定时，φ 节点需要处理形状传播：

$$\text{shape}(T_{\phi}) = \text{merge}(\text{shape}(T_{branch1}), \text{shape}(T_{branch2}))$$

合并规则：
- 静态维度必须相等
- 动态维度取最宽松的约束

### 2.4.3 内存 SSA 与别名分析

在 AI 编译器中，内存 SSA（Memory SSA）用于跟踪内存状态的演化，这对于优化内存访问至关重要。

**内存 SSA 表示**：

每个内存操作创建新的内存状态版本：
```
mem_0 = initial_memory
(val_0, mem_1) = load(addr_0, mem_0)
mem_2 = store(addr_1, val_1, mem_1)
(val_2, mem_3) = load(addr_2, mem_2)
```

**张量别名分析**：

对于带 stride 的张量，判断两个张量是否有重叠：

设张量 $A$ 和 $B$ 的访问模式为：
- $A$: base=$b_A$, shape=$(s_1^A, ..., s_n^A)$, stride=$(d_1^A, ..., d_n^A)$
- $B$: base=$b_B$, shape=$(s_1^B, ..., s_n^B)$, stride=$(d_1^B, ..., d_n^B)$

访问的地址集合：
$$\text{Addr}_A = \{b_A + \sum_{i=1}^{n} k_i \cdot d_i^A \mid 0 \leq k_i < s_i^A\}$$

别名条件：$\text{Addr}_A \cap \text{Addr}_B \neq \emptyset$

**依赖性分析**：

使用 SSA 形式简化依赖性分析：
- Read-After-Write (RAW)：真依赖
- Write-After-Read (WAR)：反依赖（SSA 中自动消除）
- Write-After-Write (WAW)：输出依赖（SSA 中自动消除）

### 2.4.4 梯度计算中的 SSA 转换

自动微分需要特殊的 SSA 处理，因为反向传播本质上是逆向的数据流：

**前向计算的 SSA**：
```
x_0 = input
y_0 = W_0 * x_0
z_0 = activation(y_0)
loss_0 = loss_fn(z_0, target)
```

**反向传播的 SSA 构建**：

反向传播需要构建对偶的 SSA 图：
```
grad_loss_0 = 1.0
grad_z_0 = d_loss_fn(z_0, target, grad_loss_0)
grad_y_0 = d_activation(y_0, grad_z_0)
grad_W_0 = x_0^T * grad_y_0
grad_x_0 = W_0^T * grad_y_0
```

**梯度累积的 SSA 表示**：

当同一参数在多处使用时，需要累积梯度：
```
grad_W_partial_1 = compute_grad_1(...)
grad_W_partial_2 = compute_grad_2(...)
grad_W_total = grad_W_partial_1 + grad_W_partial_2
```

**检查点（Checkpointing）的 SSA 影响**：

为节省内存，部分激活值不保存而是重算：
```
# 前向传播
x_0 = input
y_0 = f(x_0)  # 不保存y_0
z_0 = g(y_0)
checkpoint(x_0)  # 只保存x_0

# 反向传播
x_0' = restore_checkpoint()
y_0' = f(x_0')  # 重新计算
grad_y = backward_g(y_0', grad_z)
```

这需要 SSA 分析来确定：
1. 哪些值需要保存（活跃在反向传播中）
2. 哪些值可以重算（计算成本 vs 内存成本权衡）
3. 重算的调度顺序（最小化峰值内存）

## 2.5 MLIR 方言设计原则

Multi-Level Intermediate Representation (MLIR) 提供了一个可扩展的 IR 框架，通过方言（Dialect）机制支持不同抽象层次的表示。这对于 AI 编译器特别重要，因为需要在同一框架内处理从高级张量操作到低级硬件指令的转换。

### 2.5.1 方言的层次结构设计

**方言层次示例**：

```
┌─────────────────┐
│   Frontend      │ (tf, torch, mhlo)
├─────────────────┤
│   High-Level    │ (linalg, tensor)
├─────────────────┤
│   Mid-Level     │ (affine, scf)
├─────────────────┤
│   Low-Level     │ (llvm, gpu, spirv)
└─────────────────┘
```

**层次设计原则**：

1. **渐进式降级**：每个方言应该只降低一个抽象层次
2. **正交性**：不同方言负责不同的关注点（计算 vs 控制流 vs 内存）
3. **可组合性**：方言之间可以混合使用
4. **往返转换**：某些方言对之间应支持无损转换

**自动驾驶场景的方言设计**：

```
perception_dialect:     // 感知专用
  - conv2d_depthwise
  - nms (non-maximum suppression)
  - roi_align

planning_dialect:       // 规划专用
  - graph_search
  - trajectory_optimize
  - collision_check

control_dialect:        // 控制专用
  - pid_compute
  - kalman_filter
  - safety_check
```

### 2.5.2 类型系统与属性设计

**类型系统设计原则**：

1. **张量类型参数化**：
   ```
   tensor<?x?x?xf32, #layout_attr>
   memref<16x32xf16, affine_map<(d0, d1) -> (d1, d0)>>
   ```

2. **量化类型支持**：
   $$\text{QuantizedType} = \text{Scale} \times (\text{StorageType} - \text{ZeroPoint})$$
   
   表示为：
   ```
   !quant.uniform<i8:f32, 0.5:128>  // scale=0.5, zero_point=128
   ```

3. **稀疏类型表示**：
   ```
   #CSR = #sparse_tensor.encoding<{
     dimLevelType = ["dense", "compressed"],
     dimOrdering = affine_map<(i, j) -> (i, j)>,
     pointerBitWidth = 32,
     indexBitWidth = 32
   }>
   tensor<1024x1024xf32, #CSR>
   ```

**属性设计最佳实践**：

1. **编译时常量**：使用属性而非操作数
2. **硬件约束**：通过属性传递（如 `#gpu.block<16,16>`）
3. **优化提示**：如 `{parallel, vectorize, unroll_factor=4}`

### 2.5.3 Operation 语义定义

**操作定义的完整性要求**：

每个操作必须定义：
1. **操作数和结果类型约束**
2. **语义的数学描述**
3. **副作用标记**（纯函数、内存读写、控制流影响）
4. **规范化模式**（canonicalization patterns）
5. **合法性验证**（verification）

**示例：矩阵乘法操作定义**：

```
def MatMulOp : Op<"matmul", [NoSideEffect]> {
  let arguments = (ins 
    Tensor:$lhs,
    Tensor:$rhs,
    OptionalAttr<F32Attr>:$alpha,
    OptionalAttr<BoolAttr>:$transpose_a,
    OptionalAttr<BoolAttr>:$transpose_b
  );
  
  let results = (outs Tensor:$result);
  
  let verifier = [{
    // 验证维度匹配
    // lhs: [M, K], rhs: [K, N] -> result: [M, N]
  }];
  
  let hasCanonicalizer = 1;  // A*I = A, A*0 = 0
}
```

**语义的形式化描述**：

$$\text{MatMul}(A, B)_{ij} = \sum_{k=1}^{K} A_{ik} \cdot B_{kj}$$

带转置的版本：
$$\text{MatMul}(A, B, \text{trans}_A, \text{trans}_B)_{ij} = \sum_{k} A'_{ik} \cdot B'_{kj}$$

其中 $A' = \text{trans}_A ? A^T : A$

### 2.5.4 方言间的转换模式

**转换模式类型**：

1. **1-to-1 转换**：直接映射
   ```
   %0 = tensor.extract_slice %arg0[%i, %j][%sz1, %sz2][1, 1]
   =>
   %0 = memref.subview %arg0[%i, %j][%sz1, %sz2][1, 1]
   ```

2. **1-to-N 转换**：展开为多个操作
   ```
   %0 = linalg.matmul ins(%A, %B) outs(%C)
   =>
   scf.parallel (%i, %j) = (0, 0) to (%M, %N) step (1, 1) {
     %sum = scf.for %k = 0 to %K step 1 iter_args(%acc = 0) {
       %a = load %A[%i, %k]
       %b = load %B[%k, %j]
       %prod = mulf %a, %b
       %new_acc = addf %acc, %prod
       scf.yield %new_acc
     }
     store %sum, %C[%i, %j]
   }
   ```

3. **N-to-1 融合**：模式匹配与替换
   ```
   %0 = linalg.generic {indexing_maps = [...]} ins(%A) outs(%B) {
     ^bb0(%a: f32, %b: f32):
       %c = math.exp %a : f32
       linalg.yield %c : f32
   }
   %1 = linalg.reduce {axis = 1} ins(%0) outs(%C) {
     ^bb0(%a: f32, %b: f32):
       %c = addf %a, %b : f32
       linalg.yield %c : f32
   }
   =>
   %0 = custom.softmax ins(%A) outs(%C) {axis = 1}
   ```

**转换正确性保证**：

使用 Pattern Rewriter 框架确保：
1. **局部等价性**：每个转换保持局部语义
2. **终止性**：避免循环转换
3. **确定性**：转换顺序不影响最终结果

### 2.5.5 自定义方言的设计决策

**何时创建新方言**：

1. **领域特定抽象**：如量子计算、密码学运算
2. **硬件特定功能**：如 NPU 的特殊指令
3. **优化边界**：需要保持某些不变量的操作组

**设计检查清单**：

- [ ] 是否与现有方言功能重叠？
- [ ] 抽象层次是否合适？
- [ ] 是否容易降级到下层方言？
- [ ] 类型系统是否完备？
- [ ] 是否提供了足够的优化机会？
- [ ] 验证规则是否完整？

**案例：自动驾驶感知方言**：

```
dialect Perception {
  // 3D 检测框类型
  type BBox3D<f32[7]>  // x,y,z,w,h,d,yaw
  
  // 点云类型
  type PointCloud<n x 4 x f32>  // n points, (x,y,z,intensity)
  
  // 操作定义
  operation voxelize : (PointCloud) -> Tensor<X x Y x Z x F>
  operation pillars_encode : (PointCloud) -> Tensor<P x N x F>
  operation nms_3d : (BBox3D[], scores[]) -> (indices[])
}
```

性能建模：
$$T_{voxelize} = O(N_{points}) + O(X \times Y \times Z)$$
$$T_{nms} = O(N_{boxes}^2)$$（最坏情况）

## 2.6 本章小结

本章深入探讨了 AI 编译器中间表示设计的核心概念和实践原则。关键要点包括：

1. **多层 IR 架构**是现代 AI 编译器的基础，通过渐进式降级在不同抽象层次进行针对性优化。每层 IR 承载不同的优化职责，从高层的算法优化到低层的硬件特定优化。

2. **图 IR 与指令 IR** 各有优势，实践中常采用混合策略。图 IR 适合表达数据并行和优化机会识别，指令 IR 适合精确控制和复杂控制流。选择取决于具体的应用场景和优化目标。

3. **SSA 形式**在 AI 编译器中需要扩展以处理张量操作。张量 SSA 必须处理部分更新、内存别名和大规模数据的生命周期管理。在自动微分中，SSA 分析对于检查点策略至关重要。

4. **MLIR 方言机制**提供了构建可扩展 AI 编译器的框架。通过精心设计的方言层次、类型系统和转换模式，可以在统一框架内处理从前端到硬件的全栈优化。

5. **200T 模型编译**对 IR 设计提出了特殊要求，包括分布式表示、通信原语、内存层级感知等。IR 必须原生支持模型并行、数据并行和流水线并行的表达。

核心数学模型总结：
- IR 转换的语义等价性：$\text{Semantics}(P) = \text{Semantics}(\phi(P))$
- 内存占用模型：$M_{total} = M_{params} + M_{activations} + M_{gradients} + M_{optimizer} + M_{buffer}$
- 张量别名分析：$\text{Addr}_A \cap \text{Addr}_B \neq \emptyset$
- 编译时间复杂度：图 IR $O(|V| + |E|)$ vs 指令 IR $O(|BB| \times |I|)$

## 2.7 练习题

### 基础题（理解概念）

**练习 2.1**：多层 IR 设计
给定一个简单的神经网络层：$y = \text{ReLU}(Wx + b)$，设计三层 IR 表示（高层、中层、低层），并说明每层的优化机会。

<details>
<summary>提示</summary>
考虑高层的算子融合、中层的循环优化、低层的向量化。
</details>

<details>
<summary>参考答案</summary>

高层 IR：
```
%y = nn.linear(%x, %W, %b)
%out = nn.relu(%y)
```
优化机会：融合 linear 和 relu，消除中间结果

中层 IR：
```
for i in 0..M:
  for j in 0..N:
    sum = 0
    for k in 0..K:
      sum += W[i][k] * x[k][j]
    y[i][j] = max(0, sum + b[i])
```
优化机会：循环分块、并行化、向量化

低层 IR：
```
vmov v0, #0
vld1 {v1}, [W_ptr]
vld1 {v2}, [x_ptr]
vfma v3, v1, v2
vmax v4, v3, v0
vst1 {v4}, [y_ptr]
```
优化机会：指令调度、寄存器分配
</details>

**练习 2.2**：图 IR vs 指令 IR
一个包含动态控制流的模型：如果置信度 > 0.5 执行复杂处理，否则执行简单处理。分别用图 IR 和指令 IR 表示，并分析优缺点。

<details>
<summary>提示</summary>
图 IR 需要 Switch/Merge 节点，指令 IR 使用条件跳转。
</details>

<details>
<summary>参考答案</summary>

图 IR 表示：
```
     [Input]
        |
    [Confidence]
        |
     [Compare > 0.5]
      /        \
   [Switch]  [Switch]
    /           \
[Complex]    [Simple]
    \           /
     [Merge]
        |
     [Output]
```
优点：数据流清晰，易于识别并行机会
缺点：控制流表达复杂，需要特殊节点

指令 IR 表示：
```
  %conf = compute_confidence(%input)
  %cond = fcmp ogt %conf, 0.5
  br i1 %cond, label %complex, label %simple

complex:
  %res1 = call @complex_process(%input)
  br label %merge

simple:
  %res2 = call @simple_process(%input)
  br label %merge

merge:
  %result = phi [%res1, %complex], [%res2, %simple]
  ret %result
```
优点：控制流自然，易于实现分支预测
缺点：数据依赖不直观，优化机会难识别
</details>

**练习 2.3**：SSA 形式转换
将下面的张量操作转换为 SSA 形式，考虑内存别名：
```python
A = zeros([100, 100])
B = A[10:20, :]  # B 是 A 的视图
B = B + 1
C = A[15:25, :]  # C 与 B 有重叠
```

<details>
<summary>提示</summary>
需要跟踪内存版本，视图操作不创建新内存。
</details>

<details>
<summary>参考答案</summary>

SSA 形式：
```
A_0 = zeros([100, 100])
mem_0 = initial_memory_state
mem_1 = store(A_0, mem_0)

B_0 = view(A_0, slice=[10:20, :])  # 别名关系：B_0 aliases A_0[10:20, :]

# B = B + 1 影响底层内存
temp_0 = B_0 + 1
mem_2 = store_slice(temp_0, mem_1, base=A_0, slice=[10:20, :])
A_1 = mem_2.get_tensor(A_0.id)  # A 的新版本
B_1 = view(A_1, slice=[10:20, :])  # B 的新版本

C_0 = view(A_1, slice=[15:25, :])  # C_0 aliases A_1[15:25, :]

# 别名分析结果：
# B_1 和 C_0 有重叠：[15:20, :] 区域
```
</details>

### 挑战题（深入思考）

**练习 2.4**：渐进式降级成本模型
设计一个成本模型来决定何时进行 IR 降级。考虑编译时间 $T_c$、优化机会 $O$、代码质量 $Q$ 之间的权衡。

<details>
<summary>提示</summary>
建立数学模型：$\text{Cost} = \alpha T_c - \beta O - \gamma Q$，需要考虑不同层级的特点。
</details>

<details>
<summary>参考答案</summary>

成本模型设计：

设 IR 层级为 $L \in \{0, 1, 2, ..., n\}$，0 为最高层。

对于层级 $L$：
- 编译时间：$T_c(L) = T_{base} \cdot 2^L$（指数增长）
- 优化机会：$O(L) = O_{max} \cdot (1 - L/n)$（线性递减）
- 代码质量：$Q(L) = Q_{min} + (Q_{max} - Q_{min}) \cdot L/n$（线性递增）

总成本函数：
$$\text{Cost}(L) = \alpha T_{base} \cdot 2^L - \beta O_{max} \cdot (1 - L/n) - \gamma (Q_{min} + (Q_{max} - Q_{min}) \cdot L/n)$$

降级决策：当 $\frac{\partial \text{Cost}}{\partial L} < \theta$ 时触发降级

实际考虑因素：
1. 模型大小：大模型增加 $\alpha$（编译时间权重）
2. 部署场景：实时系统增加 $\gamma$（代码质量权重）
3. 开发阶段：调试阶段减小 $\alpha$，生产部署增加 $\gamma$
</details>

**练习 2.5**：MLIR 方言设计
为具身智能机器人设计一个操作方言，需要支持感知-决策-控制闭环，考虑实时性约束。

<details>
<summary>提示</summary>
考虑传感器融合、运动规划、安全约束等特殊操作。
</details>

<details>
<summary>参考答案</summary>

```
dialect EmbodiedAI {
  // 类型定义
  type SensorData<type, rate_hz, latency_ms>
  type Trajectory<dims, horizon, dt>
  type SafetyConstraint<type, priority>
  
  // 感知操作
  operation sensor_fusion {
    args: (camera: SensorData<image, 30, 10>,
           lidar: SensorData<pointcloud, 10, 20>,
           imu: SensorData<motion, 100, 1>)
    returns: (state: WorldState)
    attributes: {sync_policy: "latest", timeout_ms: 50}
  }
  
  // 决策操作
  operation plan_trajectory {
    args: (current: WorldState, goal: Pose, constraints: SafetyConstraint[])
    returns: (traj: Trajectory)
    attributes: {algorithm: "rrt_star", max_time_ms: 100}
  }
  
  // 控制操作
  operation compute_control {
    args: (traj: Trajectory, feedback: State)
    returns: (control: ControlCommand)
    attributes: {controller: "mpc", hz: 50}
  }
  
  // 安全检查操作
  operation safety_monitor {
    args: (state: WorldState, cmd: ControlCommand)
    returns: (safe_cmd: ControlCommand, emergency: bool)
    attributes: {redundancy: 3, fail_safe: "stop"}
  }
  
  // 实时约束表达
  operation deadline_scope {
    args: (ops: Operation[], deadline_ms: int)
    returns: (results: Any[])
    semantics: "Execute ops within deadline or trigger fallback"
  }
}
```

性能约束建模：
- 感知延迟：$T_{perception} \leq 100ms$
- 决策周期：$T_{planning} \leq 200ms$
- 控制频率：$f_{control} \geq 50Hz$
- 端到端延迟：$T_{e2e} \leq 300ms$
</details>

**练习 2.6**：张量别名分析
两个 4D 张量 A[N, C, H, W] 和 B[N', C', H', W'] 通过 stride 访问内存。设计算法判断是否存在重叠，考虑 stride 可能为负数的情况。

<details>
<summary>提示</summary>
将多维索引映射到一维地址空间，使用区间重叠判断。
</details>

<details>
<summary>参考答案</summary>

算法设计：

1. 地址计算函数：
对于张量 T 的索引 $(i_0, i_1, i_2, i_3)$：
$$\text{addr}(i_0, i_1, i_2, i_3) = base + \sum_{k=0}^{3} i_k \cdot stride_k$$

2. 地址范围计算：
考虑 stride 可能为负：
$$\text{min\_addr}_k = \begin{cases}
base + 0 & \text{if } stride_k \geq 0 \\
base + (shape_k - 1) \cdot stride_k & \text{if } stride_k < 0
\end{cases}$$

$$\text{max\_addr}_k = \begin{cases}
base + (shape_k - 1) \cdot stride_k & \text{if } stride_k \geq 0 \\
base + 0 & \text{if } stride_k < 0
\end{cases}$$

3. 总地址范围：
$$\text{Range}_T = [\sum_k \text{min\_addr}_k, \sum_k \text{max\_addr}_k]$$

4. 重叠判断：
$$\text{Overlap}(A, B) = \text{Range}_A \cap \text{Range}_B \neq \emptyset$$

5. 精确重叠检测（可选）：
如果范围重叠，进一步检查是否存在整数解：
$$\exists (i_0^A, ..., i_3^A, i_0^B, ..., i_3^B) : \text{addr}_A(...) = \text{addr}_B(...)$$

这是一个整数线性规划问题。
</details>

**练习 2.7**：梯度检查点策略
给定计算图，内存限制 M，设计最优检查点策略。每个操作有计算成本 $c_i$ 和内存占用 $m_i$。

<details>
<summary>提示</summary>
这是一个动态规划问题，需要权衡重计算成本和内存节省。
</details>

<details>
<summary>参考答案</summary>

动态规划解法：

状态定义：$dp[i][j]$ = 从操作 i 到操作 j 的最小成本

递归关系：
$$dp[i][j] = \min_{i \leq k < j} \{dp[i][k] + dp[k+1][j] + \text{merge\_cost}(k)\}$$

其中 merge_cost 考虑：
1. 内存峰值：$\max(M_{forward}[i:k], M_{forward}[k+1:j], M_{backward}[j:k+1], M_{backward}[k:i])$
2. 重计算成本：$\sum_{op \in recompute} c_{op}$

算法步骤：
1. 构建依赖图
2. 计算每个子图的内存需求
3. 枚举检查点位置
4. 选择满足内存约束的最小成本方案

优化目标函数：
$$\min \sum_{i \in checkpoints} c_i^{recompute} \\
s.t. \quad M_{peak} \leq M_{limit}$$

实际考虑：
- 优先检查点低成本高内存的操作
- 考虑操作的并行性
- 权衡编译时间和运行时性能
</details>

**练习 2.8**：200T 模型的 IR 分片表示
设计 IR 表示来支持 200T 参数模型的 3D 并行（DP=8, TP=8, PP=16）。考虑通信模式和内存布局。

<details>
<summary>提示</summary>
需要在 IR 中编码分片信息、通信操作和同步点。
</details>

<details>
<summary>参考答案</summary>

IR 设计方案：

1. 分片张量类型：
```
type ShardedTensor<
  shape: [d0, d1, ..., dn],
  dtype: type,
  sharding: {
    mesh: [DP:8, TP:8, PP:16],  // 3D mesh
    dim_sharding: [DP, TP, None, None],  // 每个维度的分片方式
    replica_groups: [[0-7], [8-15], ...]  // 副本组
  }
>
```

2. 通信原语：
```
operation all_reduce {
  args: (tensor: ShardedTensor, groups: ReplicaGroups)
  returns: (result: ShardedTensor)
  attributes: {op: "sum", async: true}
}

operation all_to_all {
  args: (tensor: ShardedTensor, 
         split_dim: int, 
         concat_dim: int)
  returns: (result: ShardedTensor)
}

operation pipeline_send_recv {
  args: (send_tensor: ShardedTensor,
         recv_shape: Shape)
  returns: (recv_tensor: ShardedTensor)
  attributes: {stage_id: int, micro_batch: int}
}
```

3. 内存估算：
每个设备的内存需求：
$$M_{device} = \frac{M_{params}}{DP \times TP \times PP} + \frac{M_{activations}}{DP} + M_{buffer}$$

对于 200T 模型：
- 参数：$\frac{400TB}{8 \times 8 \times 16} = 390GB$ per device
- 激活：取决于批大小和序列长度
- 通信缓冲：~10% 额外开销

4. 同步点表示：
```
operation sync_point {
  args: (deps: Operation[])
  attributes: {
    type: "pipeline_flush" | "gradient_sync" | "param_update",
    timeout_ms: int
  }
}
```

5. 性能模型：
通信时间：
$$T_{comm} = \frac{Data\_size}{Bandwidth} + Latency \times Hops$$

计算/通信重叠：
$$T_{total} = \max(T_{compute}, T_{comm}) + T_{sync}$$
</details>

## 2.8 常见陷阱与错误

### 陷阱 1：过早降级
**问题**：过早将高层 IR 降级到低层，丢失优化机会。
**症状**：编译后代码性能不如预期，明显的优化机会被错过。
**解决**：保持在高层 IR 尽可能长时间，只在必要时降级。

### 陷阱 2：IR 层次过多
**问题**：设计过多的 IR 层次，增加维护成本和编译时间。
**症状**：编译时间过长，调试困难，转换规则复杂。
**解决**：通常 3-4 层 IR 足够，每层应有明确的职责。

### 陷阱 3：忽视内存别名
**问题**：在 SSA 转换时未正确处理张量别名关系。
**症状**：优化后结果错误，特别是涉及 in-place 操作时。
**解决**：维护精确的别名信息，使用内存 SSA 跟踪内存状态。

### 陷阱 4：动态形状处理不当
**问题**：假设所有形状在编译时已知。
**症状**：运行时崩溃或性能严重下降。
**解决**：使用符号形状，设计形状推导规则，支持运行时特化。

### 陷阱 5：控制流与数据流混淆
**问题**：在图 IR 中强行表达复杂控制流。
**症状**：IR 结构复杂，难以理解和优化。
**解决**：使用混合 IR，控制密集区域用指令 IR。

### 陷阱 6：方言设计过于特化
**问题**：为每个小功能创建新方言。
**症状**：方言数量爆炸，转换规则组合爆炸。
**解决**：方言应该代表一个抽象层次，不是一个功能。

### 陷阱 7：忽视验证
**问题**：IR 转换缺乏正确性验证。
**症状**：间歇性错误，难以调试的问题。
**解决**：每个转换都应有验证规则，使用差分测试。

### 陷阱 8：性能模型缺失
**问题**：没有成本模型指导优化决策。
**症状**：优化可能导致性能下降。
**解决**：建立准确的性能模型，基于实际测量校准。

## 2.9 最佳实践检查清单

### IR 设计审查

- [ ] **分层合理性**
  - 每层 IR 是否有明确的抽象级别？
  - 层次之间的转换是否清晰定义？
  - 是否避免了层次过多或过少？

- [ ] **类型系统完备性**
  - 是否支持所有必要的数据类型？
  - 类型转换规则是否明确？
  - 是否支持自定义类型扩展？

- [ ] **语义保持**
  - 每个 IR 转换是否保持语义等价？
  - 是否有形式化验证或测试？
  - 数值精度损失是否在可接受范围？

### 性能考虑

- [ ] **编译时间**
  - IR 转换的时间复杂度是否合理？
  - 是否有增量编译支持？
  - 大模型编译时间是否可接受？

- [ ] **内存效率**
  - IR 表示的内存开销是否合理？
  - 是否支持内存复用分析？
  - 是否有内存泄漏风险？

- [ ] **优化机会**
  - 高层 IR 是否保留足够的语义信息？
  - 是否容易识别常见优化模式？
  - 是否支持跨层优化？

### 可扩展性

- [ ] **硬件适配**
  - 是否容易添加新硬件后端？
  - 硬件特性是否能在 IR 中表达？
  - 是否支持异构计算？

- [ ] **模型支持**
  - 是否支持主流模型结构？
  - 动态模型是否能高效处理？
  - 是否支持自定义算子？

- [ ] **并行扩展**
  - 是否原生支持各种并行模式？
  - 通信原语是否完备？
  - 是否支持大规模分布式？

### 工程实践

- [ ] **可调试性**
  - IR 是否易于理解和调试？
  - 是否有可视化工具？
  - 错误信息是否有用？

- [ ] **文档完整性**
  - 每个 IR 操作是否有清晰文档？
  - 转换规则是否有文档？
  - 是否有使用示例？

- [ ] **测试覆盖**
  - 是否有完整的单元测试？
  - 是否有端到端测试？
  - 是否有性能回归测试？

### 特殊场景

- [ ] **实时系统**
  - 是否支持确定性执行？
  - 最坏情况执行时间是否可预测？
  - 是否支持优先级调度？

- [ ] **安全关键**
  - 是否有安全性验证？
  - 是否支持故障恢复？
  - 是否有冗余检查？

- [ ] **资源受限**
  - 是否支持内存受限设备？
  - 是否有功耗优化？
  - 是否支持模型压缩？

---

*下一章：[第 3 章：计算图表示与分析](chapter3.md)*