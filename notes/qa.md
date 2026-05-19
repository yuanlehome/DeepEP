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

---

## Q5：`EventHandle` 和 `EventOverlap` 的区别是什么？

**问题**

`elastic.py` 中同时 import 了 `EventHandle`（来自 C++ 扩展）和 `EventOverlap`（来自 `deep_ep/utils/event.py`），两者有什么区别？

**答案**

### `EventHandle`（C++ 层，`csrc/utils/event.hpp`）

底层 C++ struct，直接封装 CUDA event 的核心操作：

- 持有 `std::shared_ptr<torch::Event>`，即实际的 CUDA event
- 持有 `tensors_to_record`，保持 tensor 引用计数，防止异步通信期间被提前释放
- 构造时自动在当前/指定 stream 上 `record` event
- 提供 `current_stream_wait()` 让当前 stream 等待该 event 完成

本质是一个轻量的 CUDA event 句柄。

### `EventOverlap`（Python 层，`deep_ep/utils/event.py`）

Python wrapper 类，围绕 `EventHandle` 提供更高层的使用便利性：

- 内部持有一个 `EventHandle` 实例（`self.event`）
- 额外持有 `extra_tensors` 模拟 `record_stream`（兼容 CUDA Graph）
- 实现 context manager（`__enter__`/`__exit__`），支持 `with` 语法做计算与通信 overlap
- 提供 `release_handle` 机制，退出时可选择释放底层 event 引用

### 关系总结

- `EventHandle` 是"事件本身"
- `EventOverlap` 是"用事件做 overlap 的工具类"，包装了 `EventHandle`

---

## Q6：`weak_lru` 的实现原理是什么？

**问题**

`deep_ep/utils/semantic.py` 中的 `weak_lru` 装饰器是什么？为什么不直接用 `functools.lru_cache`？

**答案**

### 问题背景

直接对实例方法使用 `@functools.lru_cache`，`self` 会被缓存强引用，导致实例永远无法被 GC 回收，造成内存泄漏。

### 实现机制

核心思路是用 `weakref.ref(self)` 替代 `self` 作为缓存的 key：

```python
def weak_lru(maxsize=128, typed=False):
    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)   # _self() 解引用 weakref

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)  # 传入弱引用

        return inner
    return wrapper
```

### 为什么能避免内存泄漏

- `weakref.ref(self)` 不增加 `self` 的引用计数
- 当实例被销毁后，弱引用失效，缓存不会阻止 GC 回收
- 对比直接 `lru_cache`：`self` 被缓存字典强引用 → 实例永远不会被 GC

### 在 DeepEP 中的用途

用于 `ElasticBuffer.get_theoretical_num_sms` 方法，缓存 SM 数估算结果。Buffer 持有大量 GPU 显存，`weak_lru` 保证缓存是 Buffer 的附属品，不会反过来控制 Buffer 的生命周期。

### 为什么 Python 标准库不提供 `weak_lru`？

- `lru_cache` 最初设计给纯函数用，方法缓存是衍生用法
- 不是所有对象都支持 `weakref`（如 `int`、`str`、部分 `__slots__` 类）
- 实例被 GC 后缓存命中的语义不统一（抛异常？重算？返回 None？）
- 20 行代码即可实现，社区认为不值得加入标准库维护负担

---

## Q7：`num_recv_tokens` 是去重前还是去重后的数？`psum` 的 offset 用于哪个对象？

**问题**

`EPHandle.num_recv_tokens` 表示的 token 数是去重前还是去重后的？`align(psum[i], expert_alignment)` 给出的 offset 是用来索引什么对象的？

**答案**

### `num_recv_tokens` 的去重语义

取决于 `do_expand` 模式：

- **非 expand 模式（`do_expand=False`）**：**去重后**的数量。同一 token 即使有多个 top-k expert 落在同一 rank，只发送/接收一次。
- **Expand 模式（`do_expand=True`）**：**去重前（展开后）**的数量。每个 token-expert 对都有独立槽位。

### `align(psum[i], expert_alignment)` 的 offset 用途

用于索引 **`recv_x` tensor**（接收到的 token hidden states），切分出每个 local expert 的输入片段：

```python
start_i = align(psum[i-1], expert_alignment)   # expert i 在 recv_x 中的起始行
end_i   = start_i + real_count[i]              # 实际有效行数
expert_i_tokens = recv_x[start_i : end_i]      # 送入 expert i 做 FFN
```

对齐是为了满足 CUDA kernel 向量化访问要求。

### `recv_x` 的真实数据形态

```
recv_x: shape = [num_recv_tokens, hidden], dtype = bf16 或 fp8
```

行按 local expert 分段排列，段间有 padding（由 `expert_alignment` 决定）。FP8 情况下 `recv_x` 是 tuple `(data, scale_factor)`。

本质上就是：从各 rank 收集来的、需要由本 rank expert 处理的 token embedding 矩阵。

---

## Q8：`__init__` 末尾的三行同步（`synchronize` → `barrier` → `synchronize`）为什么需要三步？

**问题**

`elastic.py`：

```python
torch.cuda.synchronize()
group.barrier()
torch.cuda.synchronize()
```

为什么要执行这三行？`cuda.synchronize()` 为什么要调两次？

**答案**

### 目的

确保**所有 rank 都完成了 runtime 初始化后，才允许任何 rank 发起通信**。CPU 和 GPU 是异步的，单独一次 `synchronize` 或 `barrier` 都不够。

### 逐行解释

1. **第一次 `torch.cuda.synchronize()`**：等待本地 GPU 上所有排队操作完成（如 CUDA memory 注册、NVLink mapping 等异步提交的初始化操作）。保证本 rank 的 GPU 资源确实准备好了，再告诉别人"我就绪了"。

2. **`group.barrier()`**：CPU 级别的集体同步——所有 rank 都执行完第一次 sync 后才能通过。建立全局 happens-before：通过 barrier 后，每个 rank 都可以确信所有 peer 的 GPU 资源已就绪。

3. **第二次 `torch.cuda.synchronize()`**：`group.barrier()` 内部可能使用 NCCL allreduce 实现（NCCL 操作会提交 GPU kernel）。第二次 sync 确保 barrier 的 GPU 侧工作也完成，后续 CUDA 操作不会与 barrier 内部 kernel 竞争。

### 时间线

```
① sync  → 保证本 rank GPU 初始化完成
② barrier → 保证所有 rank 都完成了 ①
③ sync  → 保证 barrier 本身的 GPU 操作也完成
─── 此后任何 rank 发起通信都是安全的 ───
```

### 为什么缺一不可

- 缺第一次 sync：可能在本地 GPU 未就绪时就声称就绪，对端 RDMA 写入写到未注册完成的内存
- 缺 barrier：无法确认其他 rank 是否就绪，可能向未初始化的 peer 发通信
- 缺第二次 sync：barrier 的 NCCL kernel 还在跑时后续操作就开始了，产生 stream 竞争

---

## Q9：FP8 量化中为什么是"每 32 个元素一个 scale factor"？

**问题**

`calculate_elastic_buffer_size` 中计算 SF 开销用的是 `ceil_div(hidden, 32)`，为什么分组粒度恰好是 32？

**答案**

### FP8 的精度局限

FP8（E4M3）只有 4 bit 指数，动态范围极窄。如果对整个 hidden 维度只用一个 scale factor，量化误差会非常大。业界通用做法是 **per-group quantization**（分组量化），每组独立维护一个 FP32 scale factor 来记录数值尺度。

### 为什么是 32

三个因素共同决定了这个粒度：

1. **与 TMA 对齐粒度匹配**：`kNumTMAAlignBytes = 32`，32 个 FP8 元素正好 32 字节，满足一次 TMA 加载的最小对齐单元。
2. **Warp 内并行**：一个 warp 32 个线程，每线程处理 1 个元素 + 共享 1 个 scale factor，无需跨 warp 通信。
3. **精度-开销折中**：overhead = `4 / (32×1) = 12.5%`，即 SF 额外字节占主数据的比例，处于可接受范围。

这实际上对应 NVIDIA 的 **microscaling（MX）格式** 中 block size = 32 的设计。

### 代码中的验证

```cpp
// csrc/elastic/buffer.hpp:603
EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);
```

该 assert 确保 SF 总字节 ≤ 主数据字节（FP8 下 `hidden_bytes = hidden * 1`），对 hidden ≥ 4 恒成立，是 sanity check。

---

## Q10：DeepEP 中 Channel 的概念如何理解？

**问题**

`kNumMaxChannelsPerSM = 8`，"每个 SM 最大 channel 数"，这里的 channel 是什么？

**答案**

### 本质定义

**一个 channel = 一个 warp = 一条独立的 token 传输流水线**。

Channel 只存在于 **hybrid 模式**（跨节点 RDMA + 节点内 NVLink 两级通信）中，解决的是跨节点通信的流水线并行问题。

### 代码中的对应关系

```cpp
// deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:56
// NOTES: a warp is a channel (different channels may share QPs)
const auto channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;
```

### 层次结构

```
一个 Kernel Block（占据 1 个 SM）
├── Notify warps（4~8 个，处理元数据信号）
├── Scaleout warps × N（N = num_channels_per_sm）
│     └── 每个 warp = 1 个 channel，负责与远端节点的 RDMA 数据收发
└── Forward warps × N
      └── 每个 warp = 1 个 channel，负责将收到的数据转发到节点内目标 GPU
```

### 每个 channel 独立维护的资源

- 独立的 **buffer 分区**：`scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx)`
- 独立的 **尾指针**（tail pointer）：追踪接收进度，无锁推进
- 独立的 **linked list**：记录 token 在 combine 阶段的路由链表

### Channel 数量的约束

```cpp
// csrc/elastic/buffer.hpp:759-779
num_channels_per_sm = min(
    (shared_memory - notify区) / token_bytes,   // shared memory 容量约束
    (32 - notify_warps) / 2,                     // warp 预算（scaleout + forward 各半）
    kNumMaxChannelsPerSM                         // 硬上限 8
);
num_channels = num_sms * num_channels_per_sm;
```

### 为什么需要 channel

- **无 channel**：一个 SM 串行处理所有跨节点 token，RDMA 延迟成为瓶颈
- **有 channel**：多个 channel 并行处理不同 token 子集，重叠通信与转发，类似多级流水线

### 与 buffer size 的关系

```cpp
// scaleout_recv_buffer 多预留 kNumMaxChannels 个 token 槽位
num_max_tokens_per_rank + kNumMaxChannels   // 1280 = 8 × 160
```

每个 channel 需要独立 buffer 区域以避免竞争，总共最多 1280 个 channel，每个需要 1 个 token 的尾部 padding。

---

## Q11：B200（Blackwell）与 H100（Hopper）的 SM/Warp 硬件配置对比

**问题**

B200 的 SM、channel、warp 等硬件配置是什么？

**答案**

### 核心规格对比

| 参数 | B200 (Blackwell) | H100 (Hopper) |
|------|------------------|---------------|
| 物理 SM 数 | 160 (2×80/die) | 132 |
| 启用 SM 数 | 148 (2×74/die) | 132 |
| Die 架构 | 双 chiplet | 单片 |
| CUDA Cores/SM | 128 | 128 |
| 最大并发 Warp/SM | **64** | 48 |
| Warp size（线程/warp） | 32 | 32 |
| 最大线程/SM | 2048 | 1536 |
| Shared Memory/SM | 228 KB | 228 KB |
| 寄存器文件/SM | 256 KB (64K×32bit) | 256 KB |
| Tensor Memory/SM | **256 KB（新增）** | 无 |
| NVLink 带宽/GPU | 1800 GB/s | 900 GB/s |
| HBM 容量 | 192 GB HBM3e | 80 GB HBM3 |
| 显存带宽 | 8 TB/s | 3.35 TB/s |
| L2 Cache | 192 MB | 50 MB |
| TDP | 1000 W | 700 W |

### 与 DeepEP 常量的对应

```cpp
static constexpr int kNumMaxSMs = 160;            // B200 物理 SM 上限（2×80）
static constexpr int kNumMaxChannelsPerSM = 8;    // 每 SM 最多 8 channel
static constexpr int kNumMaxChannels = 1280;      // 160 × 8
```

`kNumMaxSMs = 160` 取的是物理 SM 上限而非启用数 148，确保 buffer 对任何 SKU 都够用。

### B200 相比 H100 的关键优势

- **Warp 容量多 33%**（64 vs 48）：理论上可支持更多 channel/SM，但 shared memory 不变，实际瓶颈仍在 smem
- **NVLink 带宽翻倍**（1.8 TB/s vs 0.9 TB/s）：scaleup 域（节点内）通信吞吐直接翻倍
- **HBM 带宽翻倍+**（8 TB/s vs 3.35 TB/s）：buffer 读写更快，channel 不易被显存带宽卡住
- **L2 Cache 翻近 4 倍**（192 MB vs 50 MB）：workspace 元数据更容易命中 L2，减少 HBM 访问
- **HBM 容量翻倍+**（192 GB vs 80 GB）：可容纳更大的通信 buffer，支持更多 token/rank 或更大 hidden
- **Tensor Memory（TMEM）256 KB/SM**：新增的 SM 本地存储，Tensor Core 可直接读写，减少 smem 压力
- **双 chiplet 架构**：2×80 SM 通过 10 TB/s 片间互联组成单一逻辑 GPU，晶体管数从 80B 翻到 208B
- **FP4 支持**：新增 FP4 精度（20 PFLOPS），未来通信 buffer 可进一步压缩

### B200 对 DeepEP channel 机制的具体影响

| 维度 | H100 | B200 | 影响 |
|------|------|------|------|
| Warp/SM | 48 | 64 | channel 上限不受 warp 约束（瓶颈在 smem） |
| Shared Memory | 228 KB | 228 KB | channel/SM 的实际上限不变 |
| NVLink BW | 900 GB/s | 1800 GB/s | 每个 scaleup channel 的有效带宽翻倍 |
| HBM BW | 3.35 TB/s | 8 TB/s | buffer 读写吞吐翻倍，减少 channel 空等 |
| L2 | 50 MB | 192 MB | workspace 尾指针/信号等热数据更易缓存 |
