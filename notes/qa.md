# DeepEP Q&A

这个文档用于持续记录阅读 DeepEP 过程中遇到的问题与答案。

---

## Q1：`0 SM Engram / PP / CP` 的本质原因是什么？

**问题**

README 中提到 Engram、PP 基于 RDMA，CP 基于 Copy Engine，均标注为 `0 SM`。这里的 `0 SM` 本质原因是什么？

**答案**

`0 SM` 的本质是：这些路径的数据搬运主体不依赖 CUDA SM 执行拷贝循环，而是交给专用硬件完成。

- **Engram / PP（RDMA）**：通信请求发起后，数据由 NIC / RDMA 引擎直接读写 GPU 显存，SM 只承担薄控制面工作（发起请求、写 signal、等待完成）。
- **CP（Copy Engine）**：数据搬运走异步拷贝接口，由 GPU Copy Engine 完成 D2D / peer copy，而非 SM kernel 搬运。

因此 `0 SM` 更准确的含义是：大块数据传输的主体不占用 SM，SM 可留给 GEMM / 计算 kernel，从而更容易实现通信与计算重叠。

**注意**

`0 SM` 不等于零开销。仍会有 kernel launch、doorbell、signal/wait、buffer 管理等控制面开销，只是核心数据搬运不靠 SM。

---

## Q2：`cached_handle` 机制如何使用？如何判断 gating 决策保持不变？

**问题**

`dispatch` 支持 `handle` 参数，"如果提供了缓存 handle，则复用布局而不进行 CPU 同步"。这个机制如何使用？怎么判断 gating 决策保持不变？

**答案**

### 核心思想

在连续的自回归解码迭代中，同一请求的 `topk_idx`（token 路由到哪些 expert）通常不变，因为 gating 网络对相同请求往往输出相同结果。因此可以跨迭代复用上一次 dispatch 计算出的路由元数据，跳过昂贵的 CPU 同步。

### 使用方式

```python
cached_handle = None  # 初始为 None

for step in range(decode_steps):
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
        x, topk_idx, topk_weights,
        handle=cached_handle,   # 第一次为 None，后续传入上轮 handle
    )
    event.current_stream_wait()
    # ... expert compute ...
    combined_x, _, combine_event = buffer.combine(expert_output, handle=handle)
    combine_event.current_stream_wait()

    cached_handle = handle  # 保存供下一轮复用
```

当 `handle` 不为 `None` 时：
- `topk_idx` / `topk_weights` 从 handle 复用，调用时不得再传入
- `do_cpu_sync` 强制为 `False`，跳过 CPU ↔ GPU 同步
- 所有路由元数据（slot 分配、prefix sum 等）全部复用，C++ 底层跳过重新计算布局

### 如何判断 gating 决策保持不变？

DeepEP 本身不做这个判断，由调用方负责。实践中有三种策略：

- **直接假设不变**：decode 阶段请求集合固定、prefill 已结束，同一请求每步路由几乎不变，直接复用（最常见）
- **显式比较**：`torch.equal(new_topk_idx, cached_handle.topk_idx)`，若为 `False` 则置 `cached_handle = None` 触发重新计算
- **业务语义判断**：当 batch 发生变化（有请求完成或新请求加入）时，主动置 `cached_handle = None`

### 性能收益

- 跳过 CPU ↔ GPU 同步
- 跳过路由布局重计算（slot 分配、prefix sum 等）
- 适合 decode 阶段 token 数少（batch_size 量级）、迭代次数多的场景，累积收益显著

---

## Q3：`EventHandle` 和 `topk_idx_t` 分别作何用途？如何使用？

**问题**

`deep_ep` 从 C++ 扩展中导出了 `EventHandle` 和 `topk_idx_t`，它们分别是什么，怎么用？

**答案**

### `EventHandle`

`EventHandle` 是对 CUDA 事件的 C++ 封装，核心作用：
- 记录某个 CUDA stream 上的一个时间点（checkpoint）
- 让其他 stream 等待这个时间点
- 持有张量引用，防止异步通信期间张量被提前释放（兼容 CUDA Graph）

`dispatch` / `combine` 返回的 `EventOverlap` 对象内部包含一个 `EventHandle`，提供三种使用模式：

**模式一：直接同步等待**

```python
recv_x, _, _, handle, event = buffer.dispatch(
    x=x, topk_idx=topk_idx,
    async_with_compute_stream=True,
    allocate_on_comm_stream=True,
)
event.current_stream_wait()         # 手动等待通信完成
expert_output = expert_forward(recv_x)
```

**模式二：`with` 语法——计算与通信重叠**

```python
recv_x, _, _, handle, event = buffer.dispatch(
    x=x, topk_idx=topk_idx, async_with_compute_stream=True, ...
)
with event:
    some_independent_compute()      # 与通信并行执行
# 退出 with 时，当前流自动等待 event 完成
expert_output = expert_forward(recv_x)
```

**模式三：`previous_event`——让通信等前置计算**

```python
prev_event = buffer.capture()       # 在当前流上捕获事件
recv_x, _, _, handle, event = buffer.dispatch(
    x=x, topk_idx=topk_idx,
    previous_event=prev_event,      # 通信内核等此事件后再启动
    async_with_compute_stream=True,
    allocate_on_comm_stream=True,   # 使用 previous_event 时必须为 True
)
```

| 模式 | 用法 | 效果 |
|---|---|---|
| 同步等待 | `event.current_stream_wait()` | 简单，无重叠 |
| 计算通信重叠 | `with event: compute()` | 退出 `with` 时等待 |
| 通信等前置计算 | `previous_event=buffer.capture()` | 通信延迟启动 |

### `topk_idx_t`

`topk_idx_t` 是编译期可配置的整数类型别名，用于存储每个 token 的 top-k expert 路由索引：

```cpp
// 默认 64 位，即 int64_t；可通过编译宏 EP_NUM_TOPK_IDX_BITS 改为 16/32 位
using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;
```

Python 侧导出为对应的 `torch.dtype`（默认 `torch.int64`），用于在传入 `dispatch` 前对齐张量类型：

```python
import deep_ep, torch

scores = router(hidden_states)                       # [num_tokens, num_experts]
topk_weights, topk_idx = torch.topk(scores, k=2, dim=-1)

# 转换为 DeepEP 内核要求的索引类型
topk_idx = topk_idx.to(deep_ep.topk_idx_t)          # 通常为 torch.int64

# 用 -1 屏蔽不参与路由的 slot（可选）
# -1 表示该 slot 无效，对应 token 不路由到任何 expert
mask = torch.rand_like(topk_idx, dtype=torch.float) < 0.1
topk_idx.masked_fill_(mask, -1)
topk_weights.masked_fill_(topk_idx < 0, 0)

recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x=hidden_states,
    topk_idx=topk_idx,      # 必须是 deep_ep.topk_idx_t 类型
    topk_weights=topk_weights,
    ...
)
```

使用可配置宽度整数类型的目的：当 expert 总数较少（< 32768）时，可改用 `int16_t`，减少通信缓冲区占用和带宽开销。

---

## Q4：`csrc/kernels` 中的 kernels 实现与 `deep_ep/include/deep_ep/impls` 下的 kernels 实现有什么区别？

**问题**

`csrc/kernels` 中也有 kernel 相关代码，`deep_ep/include/deep_ep/impls` 下也有 dispatch、combine、barrier 等 kernel 实现。两者分别承担什么职责？

**答案**

核心区别是：`csrc/kernels` 更偏宿主侧封装、运行时入口和历史实现；`deep_ep/include/deep_ep/impls` 则是 elastic 路径下真正的 CUDA kernel 模板实现。

### `csrc/kernels/elastic/*.hpp`

这部分是 elastic kernels 的 C++ host/JIT wrapper，主要负责：

- 根据运行时参数拼接模板参数
- 生成一段包含 `#include <deep_ep/impls/*.cuh>` 的 JIT 源码
- 调用 `jit::compiler->build(...)` 编译 kernel
- 通过 `jit::launch_kernel(...)` 启动 kernel

例如 `csrc/kernels/elastic/dispatch.hpp` 中，`DispatchRuntime::generate_impl` 会根据 `num_scaleout_ranks` 选择 `dispatch.cuh` 或 `hybrid_dispatch.cuh`，并实例化对应的 dispatch / hybrid dispatch 函数模板。

### `deep_ep/include/deep_ep/impls/*.cuh`

这部分是真正的设备端 CUDA kernel 实现，包含具体通信和数据搬运逻辑，例如：

- token 到 rank / expert 的计数与 prefix sum
- slot 分配与 metadata 写入
- TMA load / store
- NCCL Gin put / get_sym_ptr
- GPU barrier、notify、epilogue 等

以 `deep_ep/include/deep_ep/impls/dispatch.cuh` 为例，里面定义的 `__global__ void dispatch_impl(...)` 才是真正执行 dispatch 的 kernel。

### `csrc/kernels/legacy` 与 `backend`

`csrc/kernels/legacy` 是 DeepEP V1 / NVSHMEM-based 的旧实现，属于另一套路径，命名空间是 `deep_ep::legacy`，和 elastic 的 JIT wrapper + `impls` 结构不同。

`csrc/kernels/backend` 则是运行时支撑层，包含 NCCL、NVSHMEM、CUDA driver 等封装，不是 dispatch / combine kernel 的主体实现。

### 总结

可以按下面方式理解：

- `csrc/kernels/elastic/*.hpp`：host 侧 JIT 封装层，负责生成、编译、启动 kernel
- `deep_ep/include/deep_ep/impls/*.cuh`：device 侧 kernel 实现层，负责真正的 GPU 执行逻辑
- `csrc/kernels/legacy`：旧版 V1 独立实现
- `csrc/kernels/backend`：通信和驱动 runtime 支撑层

