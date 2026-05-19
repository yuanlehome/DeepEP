# 专题：`calculate_elastic_buffer_size` 实现原理深度解析

## 一、功能定位与设计目标

`calculate_elastic_buffer_size` 的作用是：**在不实际分配 GPU 显存的情况下，根据 MoE 模型配置和集群物理拓扑，精确估算 elastic all-to-all 通信所需的最大 buffer 字节数**。

在 MoE（Mixture of Experts）推理/训练中，每个 token 可能被路由到多个 expert（top-k），而 expert 分布在不同 GPU 上。因此需要一个通信 buffer 来暂存 dispatch（发散）和 combine（聚合）两个阶段的中间数据。该函数的价值在于让用户可以提前规划显存，避免运行时 OOM。

---

## 二、调用入口（Python 层）

**文件**：`deep_ep/buffers/elastic.py:363-393`

```python
@staticmethod
def get_buffer_size_hint(group: dist.ProcessGroup,
                         num_max_tokens_per_rank: int, hidden: int,
                         num_topk: int = 0, use_fp8_dispatch: bool = False,
                         allow_hybrid_mode: bool = True,
                         allow_multiple_reduction: bool = True) -> int:
    """
    在不实际构造 buffer 的情况下，根据给定 MoE 配置获取推荐 buffer 大小，单位为字节。

    第一性原理说明：
        分布式通信 buffer 的大小不是只由 `x.nbytes` 决定，而是由"最多可能同时存在多少份数据"决定。
        MoE dispatch 可能让一个 token 被 top-k 复制到多个 expert，也可能产生发送/接收 metadata、对齐 padding、
        FP8 scale factor 和 hybrid 转发中间区。size hint 的作用是在真正分配昂贵 GPU buffer 前，
        先用模型形状和通信模式估算资源下界。

    参数：
        group：通信进程组。
        num_max_tokens_per_rank：每个 rank 的最大 token 数。
        hidden：每个 token 的 hidden 维度。
        num_topk：每个 token 选择的 top-k expert 数。
        use_fp8_dispatch：dispatch 是否使用 FP8。
        allow_hybrid_mode：是否启用 hybrid 模式。
        allow_multiple_reduction：combine 中是否允许多次规约。

    返回：
        size：推荐 buffer 大小，单位为字节。
    """
    return _C.calculate_elastic_buffer_size(
        get_nccl_comm_handle(group).get(),
        num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
        allow_hybrid_mode, allow_multiple_reduction)
```

pybind11 绑定位于 `csrc/elastic/buffer.hpp:1295`：
```cpp
m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
```

---

## 三、C++ 主函数实现

**文件**：`csrc/elastic/buffer.hpp:595-629`

```cpp
static int64_t calculate_buffer_size(const int64_t& nccl_comm,
                                     const int& num_max_tokens_per_rank, const int& hidden,
                                     int num_topk, const bool& use_fp8_dispatch,
                                     const bool& allow_hybrid_mode,
                                     const bool& allow_multiple_reduction) {
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

    // The worst case SF bytes must be less than the main part
    EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

    // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
    num_topk = num_topk == 0 ? 32 : num_topk;

    // Topology
    const auto [num_rdma_ranks, num_nvl_ranks] = nccl::get_physical_domain_size(nccl_comm);
    const auto [num_scaleout_ranks, num_scaleup_ranks] = nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
    const auto is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

    // Dispatch size
    const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
    const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;
    const auto num_dispatch_bytes = get_dispatch_buffer_size(
        num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,
        num_scaleout_ranks, num_scaleup_ranks,
        is_scaleup_nvlink);

    // Combine layout
    const auto num_combine_bytes = get_combine_buffer_size(
        num_max_tokens_per_rank, hidden, num_topk,
        num_scaleout_ranks, num_scaleup_ranks,
        is_scaleup_nvlink, allow_multiple_reduction);

    // Return the maximum of those layouts
    return std::max(num_dispatch_bytes, num_combine_bytes);
}
```

**核心逻辑分为三个阶段**：拓扑探测 → 分别计算 dispatch/combine 所需空间 → 取最大值。

---

## 四、阶段一：拓扑探测

**文件**：`csrc/kernels/backend/nccl.cu:58-62`

```cpp
std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}
```

### 概念解释

- **物理域**：
  - `num_rdma_ranks`：RDMA 互联的节点数（跨机）
  - `num_nvl_ranks`：单节点内通过 NVLink 互联的 GPU 数（通常为 8）

- **逻辑域**：
  - `allow_hybrid_mode = true`：启用两级通信
    - `num_scaleout_ranks = num_rdma_ranks`（跨节点 RDMA）
    - `num_scaleup_ranks = num_nvl_ranks`（节点内 NVLink）
  - `allow_hybrid_mode = false`：退化为单级通信
    - `num_scaleout_ranks = 1`
    - `num_scaleup_ranks = num_rdma_ranks * num_nvl_ranks`（全部 rank 按 scaleup 处理）

- **`is_scaleup_nvlink`**：判断 scaleup 通信是否真正走 NVLink 通路。当 `num_scaleup_ranks == num_nvl_ranks` 时成立。这影响是否需要额外的 send buffer（NVLink 可直接 RDMA 写对端 buffer，无需本地 staging）。

---

## 五、阶段二：Token 内存布局（核心数据结构）

通信 buffer 不是简单的 `token_count * hidden_bytes`。每个 token 在 buffer 中的存储是一个**对齐后的复合结构**。

### 5.1 TokenLayout 结构体

**文件**：`deep_ep/include/deep_ep/common/layout.cuh:180-250`

```cpp
struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    // NOTES: the top-k index is always 32-bit
    bool with_metadata;
    int num_topk, num_metadata_bytes;
    void* base;

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        with_metadata(with_metadata),
        num_topk(num_topk),
        // Metadata = topk_idx(int*K) + topk_weight(float*K) + [src_token_idx(int) + linked_list_idx(int*K)]
        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +
                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),
        base(base) {
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
    }

    template <bool kWithMBarrier, typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +
                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);
        return static_cast<dtype_t>(num_bytes);
    }
};
```

其中 `ptx::kNumTMAAlignBytes = 32`（定义于 `deep_ep/include/deep_ep/common/ptx.cuh:16`），这是 NVIDIA H100 TMA（Tensor Memory Accelerator）引擎要求的最小对齐粒度（32 字节 = 256 bit，对应 `LDG.256` 指令宽度）。

### 5.2 单 Token 内存布局示意

一个 Token 在 buffer 中的字节排布（无 MBarrier 时）：

```
|<--- align(hidden_bytes, 32) --->|<--- align(sf_bytes, 32) --->|<--- align(metadata_bytes, 32) --->|
|         隐层数据                 |       Scale Factor          |          路由元信息                |
```

**metadata_bytes 的组成**：
- `num_topk * sizeof(int)`：topk expert 索引
- `num_topk * sizeof(float)`：topk 路由权重
- （仅 dispatch）`sizeof(int)`：source token 全局索引
- （仅 dispatch）`num_topk * sizeof(int)`：linked list 索引（用于 channel 流水线）

### 5.3 Dispatch vs Combine 的 TokenLayout 差异

**Dispatch Token Layout**（`csrc/kernels/elastic/dispatch.hpp:209-212`）：
```cpp
static layout::TokenLayout get_dispatch_token_layout(
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
}
```
- `elem_size`：FP8 时为 1 字节，BF16 时为 2 字节
- `num_sf_packs`：FP8 时为 `ceil(hidden/32)`，BF16 时为 0
- `with_metadata = true`：dispatch 需要携带来源信息，以便接收方知道 token 从哪来

**Combine Token Layout**（`csrc/kernels/elastic/combine.hpp:109-112`）：
```cpp
static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}
```
- 固定使用 `sizeof(nv_bfloat16)` 作为 elem_size（combine 结果始终是 BF16）
- `num_sf_bytes = 0`：combine 不需要 scale factor
- `with_metadata = false`：combine 不需要来源元信息

---

## 六、阶段三-A：Dispatch Buffer 大小计算

**文件**：`csrc/elastic/buffer.hpp:529-557`

```cpp
static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                        const int& hidden, const int& num_sf_packs, const int& num_topk,
                                        const int& elem_size,
                                        const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                        const bool& is_scaleup_nvlink) {
    const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
    const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

    if (num_scaleout_ranks == 1) {
        // Direct dispatch
        const auto send_buffer_layout = layout::BufferLayout<false>(
            token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
        const auto recv_buffer_layout = layout::BufferLayout<false>(
            token_layout, num_ranks, num_max_tokens_per_rank);
        return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
    } else {
        // Hybrid dispatch
        const auto scaleup_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
        const auto scaleout_send_buffer = layout::BufferLayout<false>(
            token_layout, 1, num_max_tokens_per_rank);
        const auto scaleout_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_scaleout_ranks,
            /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);
        return scaleup_recv_buffer.get_num_bytes() +
               scaleout_send_buffer.get_num_bytes() +
               scaleout_recv_buffer.get_num_bytes();
    }
}
```

### Direct 模式（单节点 / 纯 NVLink）

```
总大小 = send_buffer + recv_buffer
```

- **send_buffer**：
  - NVLink 时 `num_ranks = 0`，大小为 0（NVLink 可 one-sided write 到对端 recv buffer，无需本地 staging）
  - 非 NVLink 时 `num_ranks = 1`，大小 = `token_bytes * 1 * num_max_tokens_per_rank`

- **recv_buffer**：
  - `num_ranks` 个 rank 都可能向本 rank 发送数据
  - 大小 = `token_bytes * num_ranks * num_max_tokens_per_rank`

### Hybrid 模式（跨节点）

两级通信需要三块独立 buffer：

```
总大小 = scaleup_recv + scaleout_send + scaleout_recv
```

- **scaleup_recv_buffer**：接收同节点内（scaleup 域）其他 GPU 发来的数据
  - `num_scaleup_ranks` 个来源 × 每个来源最多 `num_scaleout_ranks * num_max_tokens` 个 token
  - 这个倍数来自：hybrid 模式下节点内先做 intra-node gather，再做 inter-node forward

- **scaleout_send_buffer**：暂存准备发往远端节点的数据
  - 仅 1 个 staging 区 × `num_max_tokens_per_rank`

- **scaleout_recv_buffer**：接收远端节点发来的数据
  - `num_scaleout_ranks` 个来源 × 每个来源 `num_max_tokens_per_rank + kNumMaxChannels` 个 token
  - 额外的 `kNumMaxChannels`（= 8 × 160 = 1280）是 channel 流水线机制的尾部 padding 空间

---

## 七、阶段三-B：Combine Buffer 大小计算

**文件**：`csrc/elastic/buffer.hpp:559-593`

```cpp
static int64_t get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                       const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                       const bool& is_scaleup_nvlink,
                                       const bool& allow_multiple_reduction) {
    const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

    if (num_scaleout_ranks == 1) {
        // Direct combine
        const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;
        const auto send_buffer_layout = layout::BufferLayout<false>(
            token_layout, is_scaleup_nvlink ? 0 : num_ranks,
            // For single reduction cases, the maximum number of received tokens is
            // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
            num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
        const auto recv_buffer_layout = layout::BufferLayout<false>(
            token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
        return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
    } else {
        // Hybrid combine
        const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
        const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
        const auto scaleup_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
        const auto scaleout_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
        const auto scaleout_send_buffer = layout::BufferLayout<false>(
            token_layout, allow_multiple_reduction ? 1 : num_topk,
            /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
            num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));
        return scaleup_recv_buffer.get_num_bytes() +
               scaleout_send_buffer.get_num_bytes() +
               scaleout_recv_buffer.get_num_bytes();
    }
}
```

### Direct 模式

- **`allow_multiple_reduction = true`**（默认）：
  - `recv_buffer` 的 `num_ranks` 维度 = `min(num_ranks, num_topk)`
  - 含义：同一时刻最多只有 `min(num_ranks, num_topk)` 个 partial result 在飞行中（可以分多轮 reduce）
  - 这显著减少了 buffer 需求

- **`allow_multiple_reduction = false`**：
  - `recv_buffer` 的 `num_ranks` 维度 = `num_topk`
  - `send_buffer` 的 token 数 = `num_max_tokens_per_rank * num_topk`
  - 含义：必须一次性收集所有 topk 份数据才能规约，worst case 下 buffer 需求按 topk 倍数膨胀

### Hybrid 模式

三块 buffer 分别对应两级通信的接收和发送：

- **scaleup_recv_buffer**：接收节点内 combine 结果，`min(num_scaleup_ranks, num_topk)` × `num_scaleout_ranks * num_max_tokens`
- **scaleout_recv_buffer**：接收跨节点 combine 结果，`min(num_scaleout_ranks, num_topk)` × `num_max_tokens`
- **scaleout_send_buffer**：暂存待发送的 combine 数据，包含 channel 流水线 padding

---

## 八、BufferLayout 结构体 — 总量聚合

**文件**：`deep_ep/include/deep_ep/common/layout.cuh:252-312`

```cpp
template <bool kWithMBarrier>
struct BufferLayout {
    TokenLayout token_layout;
    int num_ranks;
    int num_max_tokens_per_rank;
    void* base;

    BufferLayout(const TokenLayout& token_layout,
                 const int& num_ranks,
                 const int& max_num_tokens_per_rank,
                 void* base = nullptr) :
        token_layout(token_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    int64_t get_num_bytes_per_token() const {
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    int64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    int64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }
};
```

**计算公式**：
```
buffer_bytes = num_ranks × num_max_tokens_per_rank × token_bytes_aligned
```

这是一个三层层次结构：Buffer → Rank 分区 → Token 槽位。

---

## 九、最终结果：取最大值

```cpp
return std::max(num_dispatch_bytes, num_combine_bytes);
```

**核心洞察**：Dispatch 和 Combine 在时间上是**互斥**的——一个 MoE 层先做 dispatch（token 分发到 expert），expert 计算完毕后再做 combine（结果回收）。因此两个阶段可以**复用同一块物理 buffer**，只需分配较大者即可。

---

## 十、关键常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `ptx::kNumTMAAlignBytes` | 32 | H100 TMA 引擎的硬件对齐要求（字节） |
| `kNumMaxChannelsPerSM` | 8 | 每个 SM 最大 channel 数 |
| `kNumMaxSMs` | 160 | 最大 SM 数（H100 满配） |
| `kNumMaxChannels` | 1280 | `8 × 160`，全局最大 channel 数 |
| num_topk 默认值 | 32 | 代码中有 `kNumTopk <= 32` 的广泛约束 |

---

## 十一、数值示例

假设典型配置：
- `hidden = 7168`，`num_topk = 8`，BF16 dispatch
- 8 GPU/节点 × 4 节点 = 32 ranks，`allow_hybrid_mode = true`
- `num_max_tokens_per_rank = 512`

**Token bytes（Dispatch，BF16）**：
- `num_hidden_bytes = 7168 * 2 = 14336`（已经 32B 对齐）
- `num_sf_bytes = 0`（BF16 无 scale factor）
- `num_metadata_bytes = 8*(4+4) + (1+8)*4 = 64 + 36 = 100` → align(100, 32) = 128
- `token_bytes = 14336 + 0 + 128 = 14464`

**Dispatch（Hybrid 模式）**：
- `scaleup_recv = 14464 * 8 * (4 * 512) = 14464 * 8 * 2048 = 236,978,176`
- `scaleout_send = 14464 * 1 * 512 = 7,405,568`
- `scaleout_recv = 14464 * 4 * (512 + 1280) = 14464 * 4 * 1792 = 103,612,416`
- **Dispatch Total ≈ 348 MB**

**Token bytes（Combine，BF16）**：
- `num_hidden_bytes = 7168 * 2 = 14336`
- `num_sf_bytes = 0`
- `num_metadata_bytes = 8*(4+4) = 64` → align(64, 32) = 64
- `token_bytes = 14336 + 0 + 64 = 14400`

**Combine（Hybrid，multiple_reduction=true）**：
- `scaleup_recv = 14400 * min(8,8) * (4 * 512) = 14400 * 8 * 2048 = 235,929,600`
- `scaleout_recv = 14400 * min(4,8) * 512 = 14400 * 4 * 512 = 29,491,200`
- `scaleout_send = 14400 * 1 * 4 * (512 + 1280) = 14400 * 4 * 1792 = 103,219,200`
- **Combine Total ≈ 368 MB**

**最终结果** = `max(348MB, 368MB)` ≈ **368 MB**

---

## 十二、设计决策总结

1. **Dispatch/Combine 共享 buffer**：两阶段时间互斥，取 max 而非 sum，节省近一半显存。

2. **32B TMA 对齐**：对齐粒度 `kNumTMAAlignBytes = 32`，对应 H100 TMA 引擎的 `LDG.256`（256-bit = 32 字节）加载宽度。

3. **num_topk 默认取 32**：整个代码库约束 `kNumTopk <= 32`，当用户传入 0 时取上界以保证 buffer 足够大。

4. **multiple_reduction 减少 buffer**：分多轮规约允许用更少的接收槽位，代价是增加 kernel 启动次数（延迟 vs 显存的 trade-off）。

5. **NVLink 免 send buffer**：NVLink 支持 one-sided remote write，无需本地 staging；非 NVLink 路径则需要额外的 send buffer。

6. **Channel 流水线 padding**：Hybrid 模式的 scaleout_recv 多预留 `kNumMaxChannels` 个 token 槽位，是 channel-based 流水线传输机制的尾部缓冲需要。

7. **FP8 引入 Scale Factor 开销**：FP8 dispatch 的每 32 个元素需要一个 `float` scale factor（`sf_pack_t`），增加 `ceil(hidden/32) * 4` 字节/token。

---

## 十三、调用链全景

```
Python: ElasticBuffer.get_buffer_size_hint()
  └─ C++ pybind: _C.calculate_elastic_buffer_size()
       └─ ElasticBuffer::calculate_buffer_size()
            ├─ nccl::get_physical_domain_size()     → (num_rdma_ranks, num_nvl_ranks)
            ├─ nccl::get_logical_domain_size()      → (num_scaleout_ranks, num_scaleup_ranks)
            ├─ get_dispatch_buffer_size()
            │    ├─ get_dispatch_token_layout()      → TokenLayout
            │    └─ BufferLayout<false>::get_num_bytes()
            ├─ get_combine_buffer_size()
            │    ├─ get_combine_token_layout()       → TokenLayout
            │    └─ BufferLayout<false>::get_num_bytes()
            └─ std::max(dispatch, combine)
```
