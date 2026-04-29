# DeepEP

DeepEP（DeepEveryParallel）是一个面向现代机器学习训练和推理的高性能通信库。该库目前专注于专家并行（EP）——提供高吞吐、低延迟的 all-to-all GPU 内核（MoE dispatch 和 combine），支持包括 FP8 在内的低精度格式；同时也提供用于流水线并行（PP）、上下文并行（CP）和远程内存访问（Engram）的实验性原语，所有设计都以零 SM 占用或尽可能少的 SM 占用为目标。所有内核都通过轻量级即时编译（JIT）模块在运行时编译，安装过程中无需进行 CUDA 编译。

尽管设计轻量，DeepEP 在各种配置下的性能都能达到或超过硬件带宽上限。

## 最新动态

- **V2 发布**：对专家并行进行了完整重构——相比 V1 使用少数倍的 SM 资源即可实现极致性能，同时显著支持更大的 scale-up 和 scale-out 域。V2 也已从 NVSHMEM 后端切换到更轻量的 **NCCL Gin 后端**。

### 新特性

- **完全 JIT**（即时编译）
- **NCCL Gin 后端**
  - 仅头文件且轻量
  - 能够复用现有 NCCL 通信器
- **EPv2**
  - 高吞吐和低延迟 API 统一到单一 `ElasticBuffer` 接口，并引入新的 GEMM 布局
  - 支持更大的 scale-up 与 scale-out 域（最高 EP2048）
  - 分析式计算 SM 与 QP 数量——不再需要自动调优
  - 仍然支持 hybrid 和 direct 两种模式
  - 对于类似 V3 的传统训练，在保持相同或更好性能的同时，SM 使用量从 24 降至 4 - 6
- **0 SM Engram**（基于 RDMA）
- **0 SM PP**（基于 RDMA）
- **0 SM CP**（基于 Copy Engine）

### 注意事项

- Buffer 大小消耗高于 V1
- 不再支持 0 SM RDMA 低延迟 EP
- Engram、PP 和 CP 均为实验性功能

### 仍在开发中的功能

- **弹性 GPU 与 CPU buffer**：一段连续虚拟地址空间，底层映射到 GPU 与 CPU 物理内存的混合体，从而实现完全自动、透明的 Engram 或不均衡 EP
- 利用 EP replay 处理负载不均衡，以减少中间 buffer 大小
- 用于 DP 与 TP 的 all-gather 更新和 reduce-scatter 实现

传统 V1 文档（基于 NVSHMEM）请参见 [docs/legacy.md](docs/legacy.md)。

## 性能

按照 V3 的配置，我们使用每批 8K tokens、7168 hidden 维度、top 8 experts、FP8 dispatch 和 BF16 combine 进行测试，得到如下结果：

| 架构 | NIC 类型 | 拓扑 | Dispatch 瓶颈带宽 | Combine 瓶颈带宽 | #SMs |
|--|--|--|--|--|--|
| SM90 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | CX7 | EP 8 x 4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (最高性能) |
| SM100 | N/A | EP 8 | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 (最少 #SM) |

说明：这些结果是逻辑带宽。例如，在 `EP 8 x 2` 场景下，90 GB/s 实际上包含本地 rank 流量。

与 V1 相比，**V2 峰值性能最高可达 1.3 倍，同时最多节省 4 倍 SM 数量**。

我们暂时省略了更大 EP 配置的结果，但鼓励感兴趣的用户直接进行基准测试。根据我们的内部经验，预计该内核在扩展规模后仍能持续打满硬件带宽。

V1 性能数据请参见 [docs/legacy.md](docs/legacy.md#performance)。

## 快速开始

### 环境要求

- Hopper（SM90）GPU，或其他支持 SM90 PTX ISA 的架构
- Python 3.8 及以上
- CUDA 版本
  - SM90 GPU 需要 CUDA 12.3 及以上
- PyTorch 2.10 及以上
- NCCL 2.30.4 及以上
- 节点内通信需要 NVLink
- 节点间通信需要 RDMA 网络

### 安装 NCCL 依赖

我们推荐使用 pip 安装 NCCL，这样 DeepEP 可以在 Python 环境中自动定位它。可以使用以下命令安装：

```bash
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

### 安装 NVSHMEM 依赖

DeepEP 也依赖 NVSHMEM 来支持传统方法。安装说明请参考我们的 [NVSHMEM 安装指南](docs/nvshmem.md)。

### 开发

```bash
# 构建并为 SO 文件创建符号链接
python setup.py build
# 你可以根据自己的平台修改具体的 SO 名称
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# 运行测试用例
# 注意：你可以根据自己的集群配置修改 `tests/utils/envs.py` 中的 `init_dist` 函数，
# 并启动到多个节点
python tests/elastic/test_ep.py
python tests/elastic/test_agrs.py
python tests/elastic/test_engram.py
python tests/elastic/test_pp.py
```

### 安装

```bash
python setup.py install
```

然后，在你的 Python 项目中导入 `deep_ep`，开始使用吧！

## 接口与示例

### Buffer 初始化

在 V2 中，所有 EP 操作——高吞吐和低延迟——都统一到单一 `ElasticBuffer` 接口下。可以通过直接指定 MoE 配置来初始化 buffer，并通过分析式方法计算最优的 SM 和 QP 数量。

```python
import torch
import torch.distributed as dist
from typing import Optional

from deep_ep import ElasticBuffer

# 通信 buffer（将在运行时分配）
_buffer: Optional[ElasticBuffer] = None

# 通信内核使用的 SM 数量（将在创建 buffer 时设置）
_num_comm_sms: int = 0


def get_buffer(group: dist.ProcessGroup,
               num_max_tokens_per_rank: int,
               hidden: int,
               num_topk: int,
               num_experts: int,
               use_fp8_dispatch: bool = False) -> ElasticBuffer:
    """初始化或获取用于 EP 通信的 ElasticBuffer。"""
    global _buffer, _num_comm_sms

    # 检查是否可以复用现有 buffer
    required_bytes = ElasticBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden,
        num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        return _buffer

    # 使用 MoE 配置分配新的 buffer
    # 注意：V2 buffer 大小消耗高于 V1
    _buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    # V2 会分析式计算最优 SM 数量——不再需要自动调优
    # 你也可以在 dispatch/combine 调用中手动指定 `num_sms` 来覆盖
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)

    return _buffer
```

### 在模型训练或推理预填充中的使用示例

V2 将 dispatch 和 combine API 统一到单一 `ElasticBuffer` 接口。下面的示例展示了如何在训练（包含反向传播）或推理预填充中使用它们。

```python
import torch
import torch.distributed as dist
from typing import Tuple, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int,
                     num_max_tokens_per_rank: int,
                     expert_alignment: int = 1) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    MoE dispatch：将 tokens 路由到所有 rank 上对应的 experts。
    同时支持 BF16 和 FP8（x 为 [data, scale_factors] 元组）输入。
    """
    global _buffer, _num_comm_sms

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    # `handle` 包含后续 combine 调用所需的路由元数据
    # `handle.num_recv_tokens_per_expert_list` 为 GEMM 提供每个 expert 的 token 数量
    # 使用 `event.current_stream_wait()` 在使用结果前同步计算流
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor,
                      grad_recv_topk_weights: torch.Tensor,
                      handle: EPHandle) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    """MoE dispatch 的反向传播实际上是一次 combine。"""
    global _buffer, _num_comm_sms

    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_topk_weights,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_grad_x, combined_grad_topk_weights, event


def combine_forward(x: torch.Tensor,
                    handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """MoE combine：将 expert 输出规约回其原始 rank。"""
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: EPHandle) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    """MoE combine 的反向传播实际上是一次 dispatch。"""
    global _buffer, _num_comm_sms

    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return grad_x, event
```

对于通信与计算重叠，使用 `EventOverlap` 接口管理通信流与计算流之间的依赖关系：

```python
# dispatch 后，在通信进行期间重叠执行计算
recv_x, recv_topk_idx, recv_topk_weights, handle, event = dispatch_forward(...)

# ... 在这里执行一些独立计算 ...

# 使用结果前等待通信完成
event.current_stream_wait()

# 现在可以安全使用 recv_x、recv_topk_idx、recv_topk_weights
```

### 在推理解码中的使用示例

推理解码使用同一个 `ElasticBuffer`。handle 缓存模式允许在 gating 决策保持不变时跨迭代复用路由元数据，从而避免冗余的 CPU 同步。

```python
import torch
from typing import Tuple, Optional, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def decode_dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                    topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                    num_experts: int,
                    num_max_tokens_per_rank: int,
                    cached_handle: Optional[EPHandle] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    用于推理解码的 MoE dispatch。
    如果提供了 `cached_handle`，则复用布局而不进行 CPU 同步。
    """
    global _buffer, _num_comm_sms

    if cached_handle is not None:
        # 复用缓存的 handle：跳过布局重新计算和 CPU 同步
        recv_x, _, _, handle, event = _buffer.dispatch(
            x,
            handle=cached_handle,
            num_sms=_num_comm_sms,
            async_with_compute_stream=True,
        )
        return recv_x, cached_handle.topk_idx, None, handle, event

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def decode_combine(x: torch.Tensor,
                   handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """用于推理解码的 MoE combine。"""
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event
```

### 环境变量

该库提供了一些可能有用的环境变量：

- 通用
    - `EP_BUFFER_DEBUG`：`0` 或 `1`，打印 buffer 初始化、SM 近似计算和后端调试信息，默认为 `0`
    - `EP_SUPPRESS_NCCL_CHECK`：`0` 或 `1`，抑制 NCCL 版本不匹配检查，默认为 `0`
    - `EP_AVOID_RECORD_STREAM`：`0` 或 `1`，避免对输出张量调用 `record_stream`，默认为 `0`
    - `EP_NUM_TOPK_IDX_BITS`：整数，覆盖 top-k 索引编码的位数，默认为 `0`（自动）
- 网络
    - `EP_NIC_NAME`：字符串，用于查询 NIC 属性的默认 NIC 名称，默认为 `mlx5_0`
    - `EP_OVERRIDE_RDMA_SL`：整数，覆盖用于流量隔离的 RDMA 服务级别索引
    - `EP_DISABLE_GIN`：`0` 或 `1`，禁用 NCCL Gin 后端（回退到非 Gin 路径），默认为 `0`
- JIT
    - `EP_JIT_DEBUG`：`0` 或 `1`，打印 JIT 调试信息，默认为 `0`
    - `EP_JIT_CACHE_DIR`：字符串，已编译内核的缓存目录，默认为 `$HOME/.deep_ep`
    - `EP_JIT_NVCC_COMPILER`：字符串，NVCC 编译器路径；默认为 `torch.utils.cpp_extension.CUDA_HOME`
    - `EP_JIT_CPP_STANDARD`：整数，C++ 标准版本，默认为 `20`
    - `EP_JIT_PRINT_COMPILER_COMMAND`：`0` 或 `1`，打印编译命令，默认为 `0`
    - `EP_JIT_PTXAS_VERBOSE`：`0` 或 `1`，显示详细 PTXAS 输出，默认为 `0`
    - `EP_JIT_PTXAS_CHECK`：`0` 或 `1`，断言编译后的内核不使用本地内存，默认为 `0`
    - `EP_JIT_WITH_LINEINFO`：`0` 或 `1`，为性能分析工具嵌入源码行信息，默认为 `0`
    - `EP_JIT_DUMP_ASM`：`0` 或 `1`，同时 dump PTX 和 SASS，默认为 `0`
    - `EP_JIT_DUMP_PTX`：`0` 或 `1`，dump PTX 输出，默认为 `0`
    - `EP_JIT_DUMP_SASS`：`0` 或 `1`，dump SASS 输出，默认为 `0`
- 调试和性能分析
    - `EP_GIN_GDAKI_DEBUG`：`0` 或 `1`，启用 NCCL Gin GDAKI 调试输出，默认为 `0`
    - `EP_USE_NVIDIA_TOOLS`：`0` 或 `1`，在外部 NVIDIA 工具下运行时跳过内部性能分析，默认为 `0`
    - `EP_DISABLE_BARRIER_PROFILING`：`0` 或 `1`，在基准测试中禁用基于 barrier 的通信性能分析，默认为 `0`
- 构建
    - `EP_NCCL_ROOT_DIR`：字符串，NCCL 安装目录路径；若未设置，则从 Python 环境自动检测
    - `EP_NVSHMEM_ROOT_DIR`：字符串，NVSHMEM 安装目录路径；若未设置，则从 Python 环境自动检测
    - `TORCH_CUDA_ARCH_LIST`：字符串，目标 CUDA 架构列表，例如 `"9.0"`
    - `DISABLE_SM90_FEATURES`：`0` 或 `1`，为传统方法禁用 SM90 特性，默认为 `0`
    - `DISABLE_AGGRESSIVE_PTX_INSTRS`：`0` 或 `1`，在传统方法中禁用激进的 load/store 指令，默认为 `0`

某些环境变量是**持久化**的：它们会在构建时被捕获，并作为默认值烘焙进安装包中。导入时，除非被当前环境变量覆盖，否则会自动应用这些默认值。持久化变量包括：`EP_JIT_CACHE_DIR`、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_NUM_TOPK_IDX_BITS`、`EP_NCCL_ROOT_DIR`。

更多细节请参考[测试代码](tests/elastic/test_ep.py)，或查看相应的 Python 文档。

## 网络配置

DeepEP 已在 InfiniBand 网络上经过完整测试。不过，从理论上讲，它同样兼容基于融合以太网的 RDMA（RoCE）。

### 流量隔离

InfiniBand 通过虚拟通道（VL）支持流量隔离。

为防止不同类型流量之间相互干扰，我们建议按如下方式将工作负载划分到不同虚拟通道：

- 专家并行工作负载
- 其他工作负载

对于 DeepEP V2，可以通过设置 `sl_idx` 参数或 `EP_OVERRIDE_RDMA_SL` 环境变量来控制虚拟通道分配。

### 自适应路由

自适应路由是 InfiniBand 交换机提供的一项高级路由功能，可将流量均匀分散到多条路径上。尽管自适应路由会引入额外延迟，我们仍建议在所有网络负载条件下启用它。

### 拥塞控制

拥塞控制会降低最大带宽，因此被禁用。如果某些场景中拥塞不可避免，建议将这些工作负载分配到低优先级虚拟通道。

### PCI 原子模式

如果硬件支持，建议使用以下命令设置 NIC 的 `PCI_ATOMIC_MODE`，以提升 RDMA 原子操作性能：

```bash
sudo mlxconfig -y -d mlx5_$i set PCI_ATOMIC_MODE=4
```

## 实验性分支

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
    - 移除 PyTorch 张量与通信 buffer 之间的拷贝，从而显著降低普通内核的 SM 使用量
    - 该 PR 由 **Tencent Network Platform Department** 提交
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
    - 使用低延迟协议，移除 RDMA atomic OPs 引入的额外 RTT 延迟
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
    - 使用 TMA 指令的新后端实现，以获得最小 SM 使用量并支持更大的 NVLink 域
    - 面向单 batch 场景的细粒度通信-计算重叠
    - 支持非 NVLink 环境的 PCIe 内核
    - 支持 NVFP4 数据类型
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
    - 该优化系列由 **AntGroup Network Platform Department** 提交
    - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347) 通过将通信内核执行与 NIC token 传输解耦，从 RDMA 路径中消除 SM 占用，为计算释放 SM
    - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483) 通过信号机制将 Down GEMM 计算与 Combine Send 通信重叠，以降低端到端延迟
    - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500) 使用 rail 优化转发和数据合并来优化跨节点 LL 算子通信，以降低延迟
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
    - 由 [MORI](https://github.com/ROCm/mori) 后端支持的 ROCm/AMD GPU 支持（低延迟模式）

## 社区分支

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep) - 支持在异构 GPU（如 Nvidia、AMD）和 NIC（如 EFA、Broadcom、CX7）上运行 DeepEP
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) - 在 IBRC 传输中添加 multi-QP 方案和双端口 NIC 支持
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace) - 用于高效、精准定位慢 rank 的诊断分析器
- [ROCm/mori](https://github.com/ROCm/mori) - AMD 面向性能关键型 AI 工作负载的下一代通信库（例如 Wide EP、KVCache transfer、Collectives）

## 致谢

DeepEP V2 构建于 [NCCL](https://github.com/nvidia/nccl) Gin 后端之上。感谢 @sjeaugey、@pakmarkthub、@sb17v、@xiaofanl-nvidia 以及 NCCL 团队的支持！

## 许可证

本代码仓库基于 [MIT License](LICENSE) 发布。

## 引用

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
