# DeepEP Q&A

这个文档用于持续记录阅读 DeepEP 过程中遇到的问题与答案。

## Q1：README_cn.md 里 “0 SM Engram / PP / CP” 的本质原因是什么？

**问题：**

`README_cn.md` 中写到：

- `0 SM Engram`（基于 RDMA）
- `0 SM PP`（基于 RDMA）
- `0 SM CP`（基于 Copy Engine）

这里的 `0 SM` 本质原因是什么？

**答案：**

本质原因是：这些路径的数据搬运主体不依赖 CUDA SM 去执行拷贝循环，而是交给专用硬件完成。

- **Engram / PP**：基于 RDMA。通信请求发起后，数据主要由 NIC / RDMA 引擎直接读写 GPU 显存，SM 主要承担很薄的控制面工作，例如发起请求、写 signal、等待完成。
- **CP**：基于 Copy Engine。数据搬运走 `cudaMemcpyBatchAsync` 等异步拷贝接口，由 GPU Copy Engine 完成 D2D / peer copy，而不是用 SM 跑 kernel 搬数据。
- 因此，`0 SM` 更准确地说是：大块数据传输的主体不占用 SM，SM 可以留给 GEMM / 计算 kernel，从而更容易实现通信与计算重叠。

**需要注意：**

`0 SM` 不等于完全没有 GPU/CPU 开销。它仍然会有 kernel launch、doorbell、signal/wait、buffer 管理等控制开销；只是核心数据搬运不靠 SM。

**相关代码线索：**

- Engram RDMA get：`deep_ep/include/deep_ep/impls/engram_fetch.cuh:38`
- PP RDMA put：`deep_ep/include/deep_ep/impls/pp_send_recv.cuh:153`
- CP / AGRS Copy Engine 路径：`csrc/elastic/buffer.hpp:427`

## Q2：`cached_handle` 机制如何使用？如何判断 gating 决策保持不变？

**问题：**

`README_cn.md` L289 中写到推理解码的 `decode_dispatch` 函数支持 `cached_handle` 参数，"如果提供了 `cached_handle`，则复用布局而不进行 CPU 同步"。这个机制如何使用？怎么判断 gating 决策保持不变？

**答案：**

### 核心思想

`cached_handle` 利用了推理解码的一个关键特性：在连续的自回归迭代中，每个请求的 `topk_idx`（token 被路由到哪些 expert）通常不会变化，因为 gating 网络对于相同的请求往往输出相同的路由结果。因此可以跨迭代**复用上一次 dispatch 计算出的路由元数据**，跳过昂贵的 CPU 同步。

### 使用方式

```python
cached_handle = None  # 初始为 None

for step in range(decode_steps):
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = decode_dispatch(
        x, topk_idx, topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        cached_handle=cached_handle,   # 第一次为 None，后续传入上轮 handle
    )
    event.current_stream_wait()
    # ... GEMM / expert compute ...
    combined_x, event = decode_combine(expert_output, handle)
    event.current_stream_wait()

    cached_handle = handle  # 保存供下一轮复用
```

当 `cached_handle is not None` 时（`deep_ep/buffers/elastic.py:741-748`）：
- `topk_idx` / `topk_weights` 直接从 handle 复用，调用时不得再传入
- `do_cpu_sync` 强制为 `False`，跳过 CPU ↔ GPU 同步
- `psum_num_recv_tokens_per_scaleup_rank`、`dst_buffer_slot_idx` 等所有路由元数据全部复用，C++ 底层跳过重新计算布局

### 如何判断 gating 决策保持不变？

DeepEP 本身不做这个判断，由调用方负责。实践中有几种策略：

- **直接假设不变**：decode 阶段请求集合固定、prefill 已结束，同一请求每步路由几乎不变，直接复用（最常见）
- **显式比较**：`torch.equal(new_topk_idx, cached_handle.topk_idx)`，若为 False 则置 `cached_handle = None` 触发重新计算
- **业务语义判断**：当 batch 发生变化（有请求完成或新请求加入）时，主动置 `cached_handle = None`

### 性能收益

- 跳过 CPU ↔ GPU 同步
- 跳过路由布局重计算（slot 分配、prefix sum 等）
- 适合 decode 阶段 token 数少（batch_size 量级）、迭代次数多的场景，累积收益显著

**相关代码线索：**

- EPHandle 定义：`deep_ep/buffers/elastic.py:24`
- dispatch 中 cached handle 处理逻辑：`deep_ep/buffers/elastic.py:741-757`
- 测试中 cached 路径等价性验证：`tests/elastic/test_ep.py:348-400`
