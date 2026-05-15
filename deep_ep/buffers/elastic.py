"""
ElasticBuffer 是 DeepEP 的 Python 侧高层封装。

本文件主要负责把 Python/Torch 世界中的张量、进程组、事件句柄和配置项，整理成
C++/CUDA 扩展 `deep_ep._C.ElasticBuffer` 能直接消费的参数。真正的高性能通信内核
在 `deep_ep._C` 对应的 C++/CUDA 实现中，本文件的职责包括：
    1. 管理 EP（Expert Parallelism，专家并行）dispatch/combine 的元数据句柄；
    2. 计算和缓存推荐的 buffer、SM、QP 等资源配置；
    3. 暴露 Engram、PP、AGRS 等实验性通信能力；
    4. 在 Python 层维持 CUDA stream/event 与 Torch tensor 生命周期的约束。

第一性原理视角：
    MoE 通信的根本问题是“稀疏计算带来的数据重排”：每个 token 只需要被少数 expert 处理，
    而这些 expert 分布在不同 GPU/rank 上。因此系统必须先把 token 从原始 rank 搬到 expert 所在 rank，
    等 expert 计算完成后再把结果搬回原始 rank 并按 top-k 权重合并。这个过程本质上不是普通
    all-reduce，而是带有动态路由表的数据搬运与反向还原。

    本文件把这个问题拆成三层：
    1. Python 层负责表达“要搬什么、搬到哪里、如何复用上次布局”；
    2. C++/CUDA runtime 负责把元数据转成具体 kernel、stream、buffer 和网络操作；
    3. 硬件层在 NVLink/RDMA/HBM/SM/QP 等资源之间做吞吐、延迟、并发和确定性的权衡。

    许多看似只是配置项的变量，例如 `num_sms`、`num_qps`、`expert_alignment`、`do_cpu_sync`，
    其实都来自同一个约束：通信不是免费操作，既要减少跨 GPU 字节数，又要让 GPU kernel 能以
    规整、对齐、可流水化的方式访问数据，同时避免 CPU 同步破坏异步执行。

背景术语：
    - EP dispatch：MoE 路由阶段把 token 按 top-k expert 分发到对应 rank/expert。
    - EP combine：expert 计算完成后把结果按原 token 归属 rank 规约回来。
    - NVLink：节点内 GPU 高速互联，通常用于 scale-up 通信。
    - RDMA：节点间网卡直连通信，通常用于 scale-out 通信。
    - SM：Streaming Multiprocessor，GPU 上执行 CUDA kernel 的计算单元。
    - QP：RDMA Queue Pair，RDMA 通信队列资源，数量影响吞吐和 doorbell 开销。
"""

# 标准库：`os` 读取环境变量，`math` 做组合数、乘积、向上取整等计算。
import os
import math

# PyTorch 负责张量对象、CUDA stream/event 互操作，以及分布式进程组抽象。
import torch
import torch.distributed as dist

# 类型标注用于说明 API 输入输出形态；运行时不会改变通信行为。
from typing import Callable, Optional, Tuple, Union, List, Sequence

# `contextmanager` 用于把 AGRS create/destroy 包装成 with 语法，避免异常时泄露 session。
from contextlib import contextmanager

# noinspection PyUnresolvedReferences
# `_C` 是 DeepEP 编译出的 C++/CUDA 扩展模块，承载真正的通信 buffer 和 kernel 调用。
import deep_ep._C as _C
# noinspection PyUnresolvedReferences
# EventHandle 是 C++ 扩展暴露的 CUDA event 句柄，可在 Python 层传递给后续通信 kernel 等待。
from deep_ep._C import EventHandle

# EventOverlap 把底层 event 封装成 Python 对象，便于调用方显式等待或与计算流重叠。
from ..utils.event import EventOverlap
# ceil_div/align 是底层 buffer size、alignment 计算的通用工具。
from ..utils.math import ceil_div, align
# value_or 简化默认值选择；weak_lru 为实例方法提供不阻止对象释放的缓存。
from ..utils.semantic import value_or, weak_lru
# envs 模块封装硬件能力探测和 PyTorch deterministic 配置检查。
from ..utils.envs import (
    check_fast_rdma_atomic_support,
    check_nvlink_connections, check_torch_deterministic,
    get_nvlink_gbs, get_rdma_gbs
)
# comm 模块从 torch.distributed ProcessGroup 中取出 NCCL communicator 句柄，交给 C++ 扩展复用。
from ..utils.comm import get_nccl_comm_handle


class EPHandle:
    """
    `ElasticBuffer.dispatch` 返回的通信上下文句柄。

    它保存一次 dispatch 过程中由路由拓扑决定的布局信息，例如每个 rank/expert 收到多少 token、
    每个 token 在接收 buffer 中的 slot、combine 时如何回到原始 token 顺序等。后续相同路由
    的 dispatch 可以复用该句柄，跳过 CPU/GPU 上的布局重算；combine 也必须依赖该句柄反向还原
    dispatch 的 token 流向。

    第一性原理说明：
        dispatch 不是单纯地把一个 tensor 切片发送出去，而是要解决“可逆置换”问题：
        token 在前向路由中被打散到多个 rank/expert，combine 时必须知道每个输出来自哪个原始 token、
        属于哪个 top-k 选择、是否需要与其它 expert 输出相加。只保存 `recv_x` 本身不足以反向还原，
        因为数据内容不携带来源信息；因此必须额外保存 routing metadata。

        handle 的价值还在于把“路由计算”和“数据搬运”解耦。路由布局通常比张量内容更稳定，
        例如推理或多 micro-batch 中可能重复使用相同 expert 分布。缓存布局后，下一次 dispatch
        可以把主要成本集中在搬运 token 数据，而不是反复计算 prefix sum、slot index 和链表元数据。

    属性：
        do_expand：是否使用展开布局（每个 token-expert 槽位一条记录）。
        num_experts：全局 expert 总数。
        expert_alignment：每个 local expert 接收 token 数的对齐粒度。
        num_max_tokens_per_rank：每个 rank 的最大 token 数，所有 rank 必须保持一致。
        num_sms：dispatch 使用的 SM 数，combine 默认复用该值。
        topk_idx：dispatch 克隆得到的 top-k expert 索引，形状为 `[num_tokens, num_topk]`。
        psum_num_recv_tokens_per_scaleup_rank：按 scale-up rank 统计的去重接收 token 数前缀和，
            形状为 `[num_scaleup_ranks]`。如果一个 token 的多个 top-k expert 落在同一个 rank，
            该 token 对该 rank 只计数一次。最后一个元素等于总接收 token 数。
        psum_num_recv_tokens_per_expert：按 local expert 统计的接收 token 数前缀和，计数会按
            `expert_alignment` 做 padding，形状为 `[num_local_experts]`。在非 expand 模式下，
            这是包含当前 expert 的前缀和；在 expand 模式下，`psum[i]` 等于 expert `i` 之前
            已对齐的累计数量加上 expert `i` 的真实未对齐 token 数。因此，
            `psum[i] - align(psum[i-1], expert_alignment)` 可恢复 expert `i` 的真实计数，
            `align(psum[i], expert_alignment)` 则给出 expert `i+1` 的起始 offset。
        num_recv_tokens_per_expert_list：Python list 形式的 per-expert 接收 token 数，位于 CPU 侧。
        recv_src_metadata：来源 token 索引与 buffer slot 索引。
        dst_buffer_slot_idx：dispatch 阶段写入目标 buffer 的 slot 索引。
        token_metadata_at_forward：每个 channel 上转发 token 的元数据，仅 hybrid 模式使用。
        channel_linked_list：每个 channel、每个 scale-up peer 的链表结构，仅 hybrid 模式使用。
        num_recv_tokens：接收到的 token 总数。
    """

    def __init__(self,
                 do_expand: bool,
                 num_experts: int, expert_alignment: int,
                 num_max_tokens_per_rank: int,
                 num_sms: int,
                 topk_idx: torch.Tensor,
                 num_recv_tokens_per_expert_list: list,
                 psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
                 psum_num_recv_tokens_per_expert: torch.Tensor,
                 recv_src_metadata: torch.Tensor,
                 dst_buffer_slot_idx: torch.Tensor,
                 token_metadata_at_forward: Optional[torch.Tensor],
                 channel_linked_list: Optional[torch.Tensor]):
        # 必须持有 top-k expert 索引，因为 combine 需要知道每个原始 token 曾经发给哪些 expert。
        # 这里通常使用 dispatch 返回的 clone，避免用户之后原地修改输入 `topk_idx` 导致 handle 失效。
        assert topk_idx is not None

        # 是否使用 expand 布局：普通布局按 token 去重后发送；expand 布局为每个 token-expert 选择保留独立槽位。
        self.do_expand = do_expand
        # 全局 expert 数量，用于把 expert id 映射到目标 rank/local expert。
        self.num_experts = num_experts
        # 每个 local expert 接收 token 数的对齐粒度，便于 CUDA kernel 使用向量化/分块访问。
        self.expert_alignment = expert_alignment
        # 每个 rank 允许的最大 token 数，所有 rank 必须一致，否则 buffer slot 计算会不一致。
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        # dispatch 时选择的 SM 数；combine 默认复用它，让正反向通信资源占用保持一致。
        self.num_sms = num_sms
        # `[num_tokens, num_topk]`，保存原始路由结果；combine 根据它把 expert 输出规约回原 token。
        self.topk_idx = topk_idx
        # scale-up 维度的接收 token 前缀和，用于定位每个同节点/逻辑组 peer 的接收区间。
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank
        # local expert 维度的接收 token 前缀和，用于切分 `recv_x` 中属于各 expert 的连续片段。
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        # CPU 侧 list 版本的 per-expert 接收计数，方便 Python 代码直接读取，不必访问 GPU tensor。
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        # 记录接收到的每个 slot 来源于哪个原始 token/rank，是 combine 反向路由的核心元数据。
        self.recv_src_metadata = recv_src_metadata
        # dispatch 阶段每个 token 写入目标 buffer 的 slot，下次复用 handle 时可跳过重新计算。
        self.dst_buffer_slot_idx = dst_buffer_slot_idx
        # hybrid 模式中跨 RDMA 转发 token 的元数据；直接/NVLink-only 模式下为 None。
        self.token_metadata_at_forward = token_metadata_at_forward
        # hybrid 模式中每个 channel/peer 的链表结构，用于组织跨 scale-up/scale-out 的转发关系。
        self.channel_linked_list = channel_linked_list

        # 由接收元数据长度推断总接收 token 数；如果没有 CPU sync，它可能只是异步路径下的近似/缓存值。
        self.num_recv_tokens = recv_src_metadata.shape[0]


class ElasticBuffer:
    """
    DeepEP 的弹性通信 buffer 主类。

    这个类本身不实现通信算法，而是负责初始化和持有 C++ runtime，并把 MoE/EP、Engram、
    pipeline parallel、AGRS 等不同通信模式统一映射到同一块底层 buffer。所谓 “elastic”
    指 buffer 后端和通信拓扑可扩展：当前主要面向 GPU 显存，代码中也预留了 CPU/混合内存后端。

    该弹性通信 buffer 支持：
        - 高吞吐 expert-parallel all-to-all，即 dispatch/combine，可使用 NVLink 和/或 RDMA；
        - Engram，即通过 RDMA 拉取远端 KV cache；
        - pipeline-parallel send/recv（PP），使用 NVLink；
        - all-gather reduce-scatter（AGRS），使用 NVLink。
    “Elastic” 表示底层内存形态具备扩展弹性：当前主要是 GPU-only，后续规划支持 CPU 以及
        GPU+CPU 混合后端。

    第一性原理说明：
        一个高性能通信库首先要回答三个问题：容量够不够、路径怎么走、什么时候同步。
        容量对应 `num_bytes` 和各种 size hint；路径对应 NVLink/RDMA、direct/hybrid、scale-up/scale-out；
        同步对应 CUDA stream、event、barrier 和 CPU sync。ElasticBuffer 把这些问题收敛到一个对象中，
        避免上层 MoE 代码直接面对硬件拓扑和异步生命周期。

        这里复用同一块底层 buffer 支持多种通信模式，是因为它们的共同抽象都是“在多个 rank 之间
        交换 CUDA tensor 的某些字节区间”。差异只在于路由元数据、访问顺序和同步语义：EP 是动态
        token-to-expert 路由，Engram 是按索引远端读取，PP 是相邻 rank 点对点，AGRS 是集体 gather/scatter。

    属性：
        group：通信进程组。
        rank_idx：当前 rank 索引。
        num_ranks：进程组中的 rank 数量。
        allow_hybrid_mode：是否启用多机通信的 hybrid 模式。hybrid 模式使用分层 RDMA + NVLink
            通信以获得更高带宽，并且更适合多平面/多 rail 网络。
        allow_multiple_reduction：combine 阶段是否允许多次规约。禁用后，combine epilogue 中只执行
            一次规约以获得更好精度，但可能增加数据传输量。
        prefer_overlap_with_compute：是否偏向通信与计算重叠。启用时倾向使用更少 SM。
        num_bytes：buffer 总大小，单位为字节。
        num_max_tokens_per_rank：默认每个 rank 的最大 token 数。
        num_scaleout_ranks：scale-out rank 数量。
        num_scaleup_ranks：scale-up rank 数量。
        scaleout_rank_idx：当前 rank 的 scale-out 维度索引。
        scaleup_rank_idx：当前 rank 的 scale-up 维度索引。
        num_rdma_ranks：物理 RDMA rank 数量。
        num_nvlink_ranks：物理 NVLink rank 数量。
        runtime：C++ runtime 对象。
    """

    def __init__(self,
                 group: dist.ProcessGroup,
                 # Provide `num_bytes`
                 num_bytes: Optional[int] = None,
                 # Or provide MoE settings (BF16 by default)
                 num_max_tokens_per_rank: int = 0,
                 hidden: int = 0,
                 num_topk: int = 0,
                 use_fp8_dispatch: bool = False,
                 # Configs
                 deterministic: bool = False,
                 allow_hybrid_mode: bool = True,
                 allow_multiple_reduction: bool = True,
                 prefer_overlap_with_compute: bool = True,
                 sl_idx: int = 3,
                 num_allocated_qps: int = 0,
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False):
        """
        初始化弹性通信 buffer。

        第一性原理说明：
            初始化阶段要先把所有 rank 对“同一块逻辑通信空间”的理解对齐。只要任意 rank 的 buffer
            大小、QP 数、hybrid 模式或 NCCL communicator 不一致，后续 peer-to-peer 写入就可能写到
            错误 offset、等待不存在的 peer，或者因为资源未注册完成而 hang。

            因此构造函数做的事情可以理解为：先根据模型形状估算最坏情况下需要多少字节，再检查硬件
            拓扑是否满足预期，接着把 RDMA/NVLink/NCCL/CUDA stream 等底层资源注册到 C++ runtime，
            最后用 barrier 确认所有 rank 都完成初始化。这个顺序体现的是分布式系统的基本原则：
            先建立一致的资源视图，再允许任何 rank 发起通信。

        参数：
            group：通信进程组。
            num_bytes：buffer 总大小，单位为字节；如果设置，会覆盖基于 MoE 配置的自动计算结果。
            num_max_tokens_per_rank：每个 rank 的最大 token 数，用于计算 buffer 大小。
            hidden：每个 token 的 hidden 维度。
            num_topk：每个 token 选择的 top-k expert 数。
            use_fp8_dispatch：是否启用 FP8 dispatch；启用后接收数据会是 FP8 tensor 与 scale factor 的 tuple。
            deterministic：是否使用确定性路由算法。
            allow_hybrid_mode：是否启用 hybrid 模式。
            allow_multiple_reduction：combine 中是否允许多次规约。
            prefer_overlap_with_compute：是否偏向通信与计算重叠。
            sl_idx：RDMA service level 索引，可通过 `EP_OVERRIDE_RDMA_SL` 环境变量覆盖。
            num_allocated_qps：为 RDMA 分配的 QP 数量，0 表示自动决定。
            num_cpu_timeout_secs：CPU 同步的超时时间，单位为秒。
            num_gpu_timeout_secs：GPU 操作的超时时间，单位为秒。
            explicitly_destroy：如果设为 True，需要显式调用 `destroy()` 释放资源；否则资源由析构函数释放。
        """
        # 保存 torch.distributed 进程组。DeepEP 的 rank 编号、world size 和 NCCL communicator 都来自它。
        self.group = group
        # 当前进程在 EP 通信组中的 rank；注意这不一定等于全局默认进程组 rank。
        self.rank_idx = group.rank()
        # EP 通信组总 rank 数，通常等于参与同一 MoE expert-parallel group 的 GPU 数。
        self.num_ranks = group.size()
        # hybrid 模式把节点间 RDMA 和节点内 NVLink 分层组织，适合多机多卡/多 rail 网络。
        self.allow_hybrid_mode = allow_hybrid_mode
        # combine 时是否允许分多次规约；允许可减少通信量，不允许通常数值精度更稳定。
        self.allow_multiple_reduction = allow_multiple_reduction
        # 是否偏向通信与计算重叠；为 True 时自动 SM 估计会尽量少占用 SM，给主计算 kernel 留资源。
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        # 从 ProcessGroup 中抽取 NCCL communicator 句柄，C++ runtime 用它完成 bootstrap/barrier 等协作。
        self.nccl_comm_handle = get_nccl_comm_handle(group)

        # 计算底层通信 buffer 大小。
        # 第一性原理：buffer 容量必须覆盖“最坏路由”下的临时发送区、接收区、metadata 区和可能的 FP8 scale 区。
        # 如果容量偏小，问题不会表现为 Python 异常，而可能是底层 kernel 越界或 peer 写入覆盖，因此这里宁可保守估计。
        if num_bytes is None:
            # 如果调用方没有显式给 `num_bytes`，则根据 MoE 配置估算所需空间。
            # `num_topk == 0` 被允许：底层仍可按 rank 数估一个保守尺寸，只是可能比精确 top-k 更大。
            num_bytes = _C.calculate_elastic_buffer_size(
                self.nccl_comm_handle.get(),
                num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
                allow_hybrid_mode, allow_multiple_reduction)
        # 调试开关：打印每个 rank 初始化的 buffer 大小，排查不同 rank 配置不一致的问题。
        if os.environ.get('EP_BUFFER_DEBUG', 0):
            print(f'Initializing EP elastic buffer with {num_bytes} bytes at rank EP {group.rank()}/{group.size()}')
        # 记录最终 buffer 字节数，后续 API 可用于诊断或复查容量。
        self.num_bytes = num_bytes

        # 保存默认最大 token 数；dispatch 未显式传入时会回退到该构造参数。
        self.num_max_tokens_per_rank = num_max_tokens_per_rank

        # 检查组内 GPU 的 NVLink/PCIe 拓扑是否满足高性能路径假设；异常拓扑通常会显著影响通信性能。
        check_nvlink_connections(group)

        # RDMA Service Level（SL）决定 InfiniBand/RoCE 网络中的服务等级，可通过环境变量临时覆盖。
        if 'EP_OVERRIDE_RDMA_SL' in os.environ:
            sl_idx = int(os.environ['EP_OVERRIDE_RDMA_SL'])

        # 自动决定预分配 RDMA QP 上限。QP 是 RDMA 的队列资源：越多可提高并行度，但也增加资源占用和 doorbell 开销。
        # 第一性原理：RDMA 通信需要“发送方提交 work request，网卡按 QP 顺序执行”。多个 QP 可并行推进多条数据流，
        # 但每个 QP 都消耗网卡/驱动资源；QP 太少会串行化，太多会让 doorbell、cache 和调度开销抵消收益。
        # TODO(tianr22): 后续需要结合 Engram 场景重新评估 QP 数量
        if num_allocated_qps == 0:
            # hybrid 模式需要更多 QP：每个通信 channel 可能独立使用 QP，额外的 1 个 QP 用于 notify warp。
            if self.allow_hybrid_mode:
                # 支持 fast RDMA atomic 时 notify/同步成本更低，因此 65 个通常足够；否则保守预留 129 个。
                num_allocated_qps = 65 if check_fast_rdma_atomic_support() else 129
            else:
                # direct 模式的通信 channel 更少，预留 16 个数据 QP + 1 个 notify QP。
                num_allocated_qps = 17
        # 运行期 `num_qps` 不能超过该预分配数量；dispatch/combine 中会断言检查。
        self.num_allocated_qps = num_allocated_qps

        # 创建 C++ runtime。Python 层只保存句柄；显存 buffer、CUDA stream、RDMA/NVLink 资源由扩展侧管理。
        self.explicitly_destroy = explicitly_destroy
        self.runtime = _C.ElasticBuffer(group.rank(), group.size(),
                                        self.nccl_comm_handle.get(),
                                        num_bytes,
                                        deterministic,
                                        allow_hybrid_mode,
                                        allow_multiple_reduction,
                                        prefer_overlap_with_compute,
                                        sl_idx, num_allocated_qps,
                                        num_cpu_timeout_secs, num_gpu_timeout_secs,
                                        self.explicitly_destroy)

        # 查询逻辑通信域大小。scale-out 通常表示跨节点/RDMA 维度，scale-up 通常表示节点内/NVLink 维度。
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        # 当前 rank 在逻辑 scale-out 维度上的坐标。
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        # 当前 rank 在逻辑 scale-up 维度上的坐标。
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks

        # 查询物理通信域大小，用于后续带宽模型判断有多少 RDMA rank 和 NVLink rank。
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()

        # 初始化后做 GPU/CPU 侧同步，保证所有 peer 都完成 runtime/buffer 注册后再进入后续通信。
        # 第一性原理：通信是一种跨进程副作用。若某个 rank 尚未注册好内存，另一个 rank 就开始 RDMA/NVLink 写入，
        # 发送方看到的只是“目标地址”，无法自动知道目标 runtime 是否准备好，所以必须用显式同步建立 happens-before 关系。
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()

    def destroy(self) -> None:
        """
        显式销毁 C++ runtime 并释放底层通信资源。

        只有构造时设置 `explicitly_destroy=True` 才能调用；否则资源释放交给对象析构流程。
        显式销毁适合长生命周期进程中主动回收 RDMA/NCCL/CUDA 资源。
        """
        # 防止调用方在非显式管理模式下破坏析构约定。
        assert self.explicitly_destroy

        # runtime 可能已被销毁；这里做幂等保护，避免二次 destroy。
        if self.runtime is not None:
            # 通知 C++ 扩展释放 buffer、stream、QP 等资源。
            self.runtime.destroy()
            # 置空后该 Python 对象不可再用于通信，避免悬挂 C++ 句柄被误用。
            self.runtime = None  # 销毁后不可再使用
            # NCCL communicator wrapper 也不再需要，帮助 Python GC 回收引用。
            self.nccl_comm_handle = None

    @staticmethod
    def get_buffer_size_hint(group: dist.ProcessGroup,
                             num_max_tokens_per_rank: int, hidden: int,
                             num_topk: int = 0, use_fp8_dispatch: bool = False,
                             allow_hybrid_mode: bool = True,
                             allow_multiple_reduction: bool = True) -> int:
        """
        在不实际构造 buffer 的情况下，根据给定 MoE 配置获取推荐 buffer 大小，单位为字节。

        第一性原理说明：
            分布式通信 buffer 的大小不是只由 `x.nbytes` 决定，而是由“最多可能同时存在多少份数据”决定。
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

    @staticmethod
    def get_engram_storage_size_hint(num_entries: int, hidden: int,
                                     num_max_tokens_per_rank: int,
                                     dtype: torch.dtype = torch.bfloat16) -> int:
        """
        （实验性）获取 Engram storage 所需的最小 buffer 大小。

        第一性原理说明：
            Engram 可理解为“分布式 KV/embedding 表的远端读取”。读取前必须为两类空间留容量：
            一类是长期存在的 storage entry，另一类是每次 fetch 暂存远端返回结果的接收槽。
            低精度 dtype 还需要 scale factor，否则仅有量化后的字节无法还原数值尺度。

        参数：
            num_entries：Engram storage 中的 entry 数量。
            hidden：每个 entry 的 hidden 维度。
            num_max_tokens_per_rank：每个 rank 的最大 token 数，用作接收空间预留。
            dtype：数据类型，默认为 `torch.bfloat16`。

        返回：
            size：推荐的 Engram storage 大小，单位为字节。
        """
        # TODO: 重构所有 API，以支持更高并行度
        # TODO: 考虑 FP4 场景
        # 低精度类型（例如 FP8，itemsize <= 1）通常还需要额外保存 scale factor；每 32 个 hidden 元素一组。
        num_sf_packs = ceil_div(hidden, 32) if dtype.itemsize <= 1 else 0
        # 每条 entry 的数据字节数 = hidden 数据本体 + scale factor 字节数；按 32B 对齐以匹配 LDG.256 访问粒度。
        num_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)
        # Engram buffer 同时容纳持久 storage entry 和最多 `num_max_tokens_per_rank` 个 fetch 接收槽。
        return num_bytes_per_entry * (num_entries + num_max_tokens_per_rank)

    @staticmethod
    def get_pp_buffer_size_hint(num_max_tensor_bytes: int,
                                num_max_inflight_tensors: int) -> int:
        """
        （实验性）获取 pipeline-parallel（PP）send/recv 所需的最小 buffer 大小。

        第一性原理说明：
            PP 通信的基本形态是环上相邻 rank 之间的点对点传输。为了让前向/反向或多个 micro-batch
            同时在途，buffer 不能只容纳一个 tensor，而要容纳“最大 tensor 字节数 × 最大在途数 × 方向数”。
            send/recv 以及 prev/next 两个邻居共同决定了这里的 2 × 2。

        参数：
            num_max_tensor_bytes：每次 send/recv 操作允许的最大 tensor 大小，单位为字节。
            num_max_inflight_tensors：同一时刻允许在途的最大 tensor 数量。

        返回：
            size：推荐的 PP buffer 大小，单位为字节。
        """
        # PP buffer 中每个 tensor slot 按 32B 对齐，便于底层 kernel 使用 LDG.256 等向量化加载。
        num_max_tensor_bytes = align(num_max_tensor_bytes, 32)

        # 需要同时预留 send/recv 两类 buffer（*2），并分别面向环形拓扑中的前驱和后继 rank（再 *2）。
        return num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2

    @staticmethod
    def get_agrs_buffer_size_hint(group: dist.ProcessGroup,
                                  num_max_session_bytes: int) -> int:
        """
        （实验性）获取 all-gather reduce-scatter（AGRS）session 所需的最小 buffer 大小。

        第一性原理说明：
            AGRS 的核心是让每个 rank 都能看到一段按 rank 排列的共享结果区。与 EP 不同，它没有
            token-level 动态路由，因此容量下界主要由单个 session 内所有 gathered tensor 的总字节数决定。
            更复杂的 offset 划分和同步由 session 配置及 runtime 管理。

        参数：
            group：通信进程组。
            num_max_session_bytes：单个 session 中所有 gathered tensor 的最大总字节数。

        返回：
            size：推荐的 AGRS buffer 大小，单位为字节。
        """
        return num_max_session_bytes

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        """
        在所有 rank 之间执行 GPU 级 barrier，可选 CPU 同步。

        第一性原理说明：
            CUDA kernel 和通信操作默认是异步排队的；CPU 代码继续执行并不代表 GPU 上的写入已经完成。
            barrier 的目的不是“等待 Python 函数返回”，而是在所有 rank 的 GPU 工作流之间建立一个共同进度点，
            确保某些跨 rank 可见的副作用已经发生。

        参数：
            use_comm_stream：是否使用通信 stream；否则使用当前计算 stream。
            with_cpu_sync：是否在 barrier 前后额外调用 `cudaDeviceSynchronize`。
        """
        self.runtime.barrier(use_comm_stream, with_cpu_sync)

    @staticmethod
    def _unpack_handle(handle: Optional[EPHandle] = None) \
        -> Tuple[Optional[int], Optional[list],
                 Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        把可选的 `EPHandle` 拆成 runtime.dispatch 所需的缓存参数。

        第一性原理说明：
            C++ runtime 只认识固定位置的参数列表，不理解 Python 对象的语义。这个函数相当于把
            “面向人的结构化对象”展开成“面向 kernel/runtime 的扁平参数”。这种展开必须保持顺序稳定，
            因为底层会按位置解释每个 tensor 是 prefix sum、slot index 还是 hybrid metadata。

        handle 为 None 表示首次 dispatch，需要底层重新计算布局；非 None 表示复用之前保存的布局元数据。
        返回值顺序必须与 `self.runtime.dispatch(...)` 的 cached 参数顺序保持一致。
        """
        # 首次 dispatch 没有可复用元数据，传 None 让 C++ runtime 走完整布局计算流程。
        if handle is None:
            return None, None, None, None, None, None, None
        # 复用 token 总数、per-expert 计数、前缀和、目标 slot、hybrid 转发元数据等。
        return (handle.num_recv_tokens,
                handle.num_recv_tokens_per_expert_list,
                handle.psum_num_recv_tokens_per_scaleup_rank,
                handle.psum_num_recv_tokens_per_expert,
                handle.dst_buffer_slot_idx,
                handle.token_metadata_at_forward,
                handle.channel_linked_list)

    @staticmethod
    def capture() -> EventHandle:
        """
        在当前 stream（即 `torch.cuda.current_stream()`）上捕获一个 CUDA event。

        第一性原理说明：
            CUDA stream 是有序队列，但不同 stream 之间默认无序。event 是跨 stream 建立依赖的最小同步原语：
            它记录“当前 stream 到此为止的工作已经排队/完成到某个点”，后续通信 stream 可以等待该 event，
            从而避免全局同步，又能保证生产者数据先于消费者通信发生。

        返回：
            event_handle：捕获到的 event 句柄。
        """
        # 构造 EventHandle 时会在当前 CUDA stream 上记录事件，供后续 DeepEP kernel 建立依赖关系。
        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """
        获取通信 stream。

        第一性原理说明：
            把通信放在独立 stream 上，是为了让 GPU 同时推进计算 kernel 和通信 kernel。只要二者没有数据依赖，
            它们就不需要排在同一个队列中串行执行；真正的依赖通过 event 显式表达。

        返回：
            stream：通信 stream。
        """
        # C++ runtime 返回的是底层通信 stream 的轻量描述，这里重新包装成 PyTorch 可识别的 CUDA Stream 对象。
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

    def get_physical_domain_size(self) -> Tuple[int, int]:
        """
        获取物理通信域大小，即 RDMA rank 数和 NVLink rank 数。

        第一性原理说明：
            物理通信域描述“硬件真实怎么连”。NVLink 通常延迟低、带宽高但只覆盖节点内；RDMA 覆盖节点间，
            但链路、网卡和交换网络开销不同。算法如果忽略物理域，把所有 rank 当作等价节点，就会在跨节点
            通信上付出过高代价。

        返回：
            num_rdma_ranks：物理 RDMA rank 数量。
            num_nvlink_ranks：物理 NVLink rank 数量。
        """
        # 物理域由底层 runtime 根据硬件拓扑/NCCL bootstrap 信息判定。
        return self.runtime.get_physical_domain_size()

    def get_logical_domain_size(self) -> Tuple[int, int]:
        """
        获取逻辑通信域大小，即 scale-out rank 数和 scale-up rank 数。

        第一性原理说明：
            逻辑通信域描述“算法希望怎么分层”。scale-up 聚合同节点内的快速互联，scale-out 处理跨节点扩展。
            这是一种把全连接 all-to-all 拆成层次化通信的方式：先在快链路内组织数据，再通过慢链路跨节点交换，
            最后在目标节点内分发，从而减少对最慢链路的压力。

        返回：
            num_scaleout_ranks：逻辑 scale-out rank 数量。
            num_scaleup_ranks：逻辑 scale-up rank 数量。
        """
        # 逻辑域用于算法分层：scale-up 负责节点内快速聚合，scale-out 负责节点间扩展。
        return self.runtime.get_logical_domain_size()

    def engram_write(self, storage: torch.Tensor) -> None:
        """
        （实验性）将 Engram storage 数据写入 buffer。
        该调用会在写入前后执行 barrier，确保所有 peer 可见。

        第一性原理说明：
            远端读取只有在“被读取的数据已经放到远端可访问内存中”时才有意义。Engram write 的本质是发布数据：
            把本 rank 的 storage 放入 runtime 管理的可寻址区域，并通过 barrier 建立可见性边界，避免其它 rank
            在数据尚未完成写入或注册前发起 RDMA get。

        参数：
            storage：Engram storage tensor，形状为 `[num_entries, hidden]`，类型为 `torch.bfloat16`。
        """
        # 当前 Engram 写入路径主要面向 BF16 storage；FP8/更低精度的 scale factor 处理仍待扩展。
        # runtime 内部会把 `storage` 拷贝/注册到 ElasticBuffer 管理的远端可访问区域。
        self.runtime.engram_write(storage)

    def engram_fetch(self, indices: torch.Tensor, num_qps: int = 0) -> Callable:
        """
        （实验性）通过 RDMA 从远端 rank 拉取 Engram entry。
        返回一个 callable；调用该 callable 时会等待 RDMA get 完成，并返回拉取到的 tensor。

        第一性原理说明：
            RDMA get 是“读取方主动拉取远端内存”的操作。与发送方主动 push 不同，它要求读取方知道远端地址、
            索引到 entry 的映射以及本地接收位置。返回 hook 的设计来自异步 I/O 原理：先发起网络请求，
            让 GPU/网卡在后台推进；等计算真正需要结果时再调用 hook 等待，从而隐藏通信延迟。

        参数：
            indices：要拉取的 entry 索引，形状为 `[num_tokens]`，类型为 `torch.int`。
            num_qps：使用的 QP 数量，0 表示使用所有已分配 QP。

        返回：
            hook：一个 callable，会阻塞直到数据到达，并返回形状为 `[num_tokens, hidden]`、
                类型为 `torch.bfloat16` 的 tensor。
        """
        # 返回 hook 而不是立即阻塞：调用方可以先发起 RDMA get，再把等待动作推迟到真正需要结果时。
        return self.runtime.engram_fetch(indices, num_qps)

    def pp_set_config(self, num_max_tensor_bytes: int, num_max_inflight_tensors: int):
        """
        （实验性）配置 pipeline-parallel（PP）send/recv 参数。该调用包含 barrier，用于 flush 之前的操作。

        第一性原理说明：
            PP 的吞吐来自流水线：多个 micro-batch 在相邻 stage 之间交错传递。为了避免新配置覆盖旧配置下
            仍在飞行的 buffer slot，修改容量/在途数前必须先 flush 旧操作。否则两个不同语义的 tensor
            可能复用同一段通信缓冲区。

        参数：
            num_max_tensor_bytes：每次 send/recv 操作允许的最大 tensor 大小，单位为字节。
            num_max_inflight_tensors：同一时刻允许在途的最大 tensor 数量。
        """
        # PP 配置会影响底层环形 send/recv buffer 的 slot 划分，runtime 内部会同步清理旧的飞行中操作。
        self.runtime.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)

    def pp_send(self, t: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:
        """
        （实验性）向 PP ring 中的相邻 rank 发送 tensor，只允许前驱或后继 rank。

        第一性原理说明：
            pipeline parallel 的数据依赖是局部的：一个 stage 的输出只会成为相邻 stage 的输入。
            因此它不需要全局 all-to-all，限制为前驱/后继可以简化 buffer 编址和同步协议，也让底层更容易
            通过固定 ring 拓扑优化 NVLink 传输。

        参数：
            t：待发送 tensor，必须连续，并且大小不超过 `num_max_tensor_bytes`。
            dst_rank_idx：目标 rank 索引，必须是 ring 中的前驱或后继 rank。
            num_sms：使用的 SM 数量，0 表示使用全部 SM。
        """
        # 发送只允许面向 PP ring 的相邻 rank，底层通过 NVLink symmetric buffer 完成点对点传输。
        self.runtime.pp_send(t, dst_rank_idx, num_sms)

    def pp_recv(self, t: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:
        """
        （实验性）从 PP ring 中的相邻 rank 接收 tensor，只允许前驱或后继 rank。

        第一性原理说明：
            接收方预先提供输出 tensor，是为了让底层通信直接写入最终位置，避免“先收到临时 buffer 再拷贝”
            的额外 HBM 流量。对于流水线并行，节省一次内存读写往往和减少网络字节数同样重要。

        参数：
            t：用于接收数据的输出 tensor，必须连续，并且大小不超过 `num_max_tensor_bytes`。
            src_rank_idx：来源 rank 索引，必须是 ring 中的前驱或后继 rank。
            num_sms：使用的 SM 数量，0 表示使用全部 SM。
        """
        # 接收 tensor `t` 由调用方预分配，runtime 将远端数据写入该输出 buffer。
        self.runtime.pp_recv(t, src_rank_idx, num_sms)

    def create_agrs_session(self) -> None:
        """
        （实验性）开始新的 all-gather reduce-scatter（AGRS）session，必须与 `destroy_agrs_session` 配对使用。

        第一性原理说明：
            session 是一段通信生命周期边界。它告诉 runtime：接下来的一组 gather/scatter 操作共享同一套
            临时 buffer、offset 分配和同步状态。没有 session 边界，runtime 很难判断什么时候可以安全复用
            中间空间。
        """
        # 创建 AGRS session 后，后续 all_gather/reduce_scatter 类操作共享同一段临时 symmetric buffer。
        self.runtime.create_agrs_session()

    def destroy_agrs_session(self) -> None:
        """
        （实验性）结束当前 AGRS session。该操作会等待 compute stream，并向所有 peer 通知 session 完成。

        第一性原理说明：
            结束 session 的关键不是释放 Python 对象，而是确认所有依赖该 session buffer 的 GPU 工作都已经越过
            安全点。否则后续 session 复用同一 buffer 时，可能覆盖仍被其它 stream 或 peer 读取的数据。
        """
        # 销毁 session 会等待/通知所有 peer，确保 session 内通信完成后再释放临时状态。
        self.runtime.destroy_agrs_session()

    @contextmanager
    def agrs_new_session(self, enabled: bool = True):
        """
        （实验性）封装 `create_agrs_session` 和 `destroy_agrs_session` 的上下文管理器。

        第一性原理说明：
            session 必须成对出现，这属于资源生命周期约束。用上下文管理器表达这个约束，可以让异常路径也执行
            `destroy_agrs_session`，避免因为中途报错导致底层认为 session 仍然活跃。

        参数：
            enabled：如果为 `False`，该上下文管理器不执行任何操作。
        """
        # enabled=False 时保留 with 语法但不创建 session，便于调用方用同一段代码开关 AGRS 优化。
        if not enabled:
            yield
            return

        # 正常路径：进入 with 时创建 session，离开 with 时无论是否异常都销毁 session。
        self.runtime.create_agrs_session()
        try:
            yield
        finally:
            self.runtime.destroy_agrs_session()

    def agrs_set_config(self, num_max_session_bytes: int,
                        num_max_all_gathers_per_session: int) -> None:
        """
        （实验性）配置 AGRS session 参数。该调用包含 barrier，用于 flush 之前的操作。

        第一性原理说明：
            AGRS 需要提前知道 session 最大容量和操作次数，才能把一段连续 buffer 划分成不重叠的子区间。
            这类似手动内存分配器：如果不先给出上界，就无法在无锁/低同步开销路径中快速计算每次 gather 的地址。

        参数：
            num_max_session_bytes：每个 session 中 gathered tensor 的最大总字节数。
            num_max_all_gathers_per_session：每个 session 中 all-gather 操作的最大次数。
        """
        # 配置 session 总容量和 all-gather 次数上限，底层据此划分每次 gather 的 buffer offset。
        self.runtime.agrs_set_config(num_max_session_bytes, num_max_all_gathers_per_session)

    # noinspection PyTypeChecker
    def agrs_get_inplace_tensor(self,
                                shapes: Union[Tuple[int, ...], torch.Size, Sequence[Union[Tuple[int, ...], torch.Size]]],
                                dtype: torch.dtype) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        （实验性）从 AGRS buffer 中获取当前 rank slot 对应的 in-place tensor，不发生拷贝。
        必须在活跃的 AGRS session 内调用。

        第一性原理说明：
            in-place tensor 的本质是“把通信 buffer 直接暴露成 PyTorch tensor view”。这样上层计算写入 tensor
            时，数据已经位于 all-gather 需要读取的位置，避免一次从普通 tensor 到通信 buffer 的拷贝。
            代价是调用方必须遵守 session 生命周期和 shape/dtype 对齐约束。

        参数：
            shapes：要分配的 tensor 形状。可以传单个 shape tuple，也可以传一组 shape tuple 进入 batched 模式。
            dtype：tensor 的数据类型。

        返回：
            tensor：如果传入单个 shape，则返回单个 tensor；batched 模式下返回 tensor tuple。
        """
        # `shapes` 既可传单个 shape，也可传多个 shape；通过首元素是否为 tuple 判断是否 batched。
        is_batched_mode = isinstance(shapes[0], tuple)
        # 非 batched 情况统一包装成 tuple，后续逻辑只处理一种结构。
        if not is_batched_mode:
            shapes = (shapes, )
        # runtime 只需要每个 tensor 的字节数；返回的是当前 rank 在 symmetric buffer 中对应的裸 byte tensor view。
        tensors = self.runtime.agrs_get_inplace_tensor(
            (math.prod(shape) * dtype.itemsize for shape in shapes)
        )
        # 把 byte view 重新解释成调用方期望的 dtype 和 shape；不发生拷贝，因此写入该 tensor 会直接写入 AGRS buffer。
        out = tuple(tensor.view(dtype).view(shape) for tensor, shape in zip(tensors, shapes, strict=True))
        # 保持 API 友好性：单输入返回单 tensor，多输入返回 tuple。
        return out if is_batched_mode else out[0]

    def all_gather(self, t: Union[torch.Tensor, Sequence[torch.Tensor]]):
        """
        （实验性）在活跃的 AGRS session 中执行 all-gather 操作。
        每个 rank 的数据会通过 NVLink symmetric memory 收集到所有 rank。

        第一性原理说明：
            all-gather 的目标是让每个 rank 拥有所有 rank 的输入副本。使用 symmetric memory 的关键在于：
            每个 rank 对同构地址/slot 有一致理解，底层可以直接把本 rank 数据写入其它 rank 对应位置。
            返回 handle 而不是立即等待，仍然是为了让数据传输与后续计算尽可能重叠。

        参数：
            t：待 all-gather 的单个 tensor 或 tensor 序列。每个 tensor 都必须连续、分配在 CUDA 上，
                并且 `nbytes` 按 32 字节对齐。

        返回：
            单 tensor 输入：返回 `(gathered, handle)`，其中 `gathered` 额外增加一个 `num_ranks` 前导维，
                `handle` 是用于等待数据到达的 callable。
            序列输入：返回 `(*gathered_tensors, handle)`，每个输入对应一个 gathered tensor。
        """
        # 单 tensor 路径：runtime 统一按 tuple 接口处理，这里把结果拆回更自然的 `(tensor, handle)`。
        if isinstance(t, torch.Tensor):
            tensors, handle = self.runtime.all_gather((t,))
            return tensors[0], handle

        # 批量路径：一次 session 内收集多个 tensor，最后一个返回值是等待数据到达的 handle/callback。
        tensors, handle = self.runtime.all_gather(t)
        return *tensors, handle

    @weak_lru(maxsize=None)
    def get_theoretical_num_sms(self, num_experts: int, num_topk: int,
                                num_scaleout_topk: int = 0,
                                rdma_gbs: float = 0, nvlink_gbs: float = 0,
                                # TODO: 针对不同 GPU 架构使用不同默认值
                                sm_read_gbs: float = 200, sm_write_gbs: float = 50) -> int:
        """
        基于带宽模型估算 dispatch/combine kernel 的最优 SM 数。
        结果会被缓存；该估算假设 gate 分布均衡。

        第一性原理说明：
            通信 kernel 占用 SM 的数量不是越多越好。SM 太少时，GPU 无法足够快地搬运/打包数据，链路吃不满；
            SM 太多时，会抢占主计算 kernel 的资源，降低通信与计算 overlap 的收益。这个函数用一个简化带宽模型
            比较 HBM 读写需求、NVLink/RDMA 链路带宽和拓扑命中概率，估算“刚好能喂饱瓶颈链路”的 SM 数。

            这里使用期望 top-k 而不是真实路由，是因为该函数在 dispatch 前调用，尚不知道本 batch 的精确分布。
            因此它用 balanced gate 假设做静态近似：把 expert 均匀分布到 group，估计一个 token 平均会触达多少 group。

        参数：
            num_experts：全局 expert 总数。
            num_topk：每个 token 选择的 top-k expert 数。
            num_scaleout_topk：为 balanced gate 预留的参数，目前必须为 0。
            rdma_gbs：RDMA 带宽，单位 GB/s；0 表示自动探测。
            nvlink_gbs：NVLink 带宽，单位 GB/s；0 表示自动探测。
            sm_read_gbs：每个 SM 的 HBM 读带宽，单位 GB/s。
            sm_write_gbs：每个 SM 的 HBM 写带宽，单位 GB/s。

        返回：
            num_sms：推荐的 SM 数，偶数且至少为 4。
        """
        # TODO: 支持 `do_expand` 和 `allow_multiple_reduction`

        # 该模型把 “1” 作为单位 token 数据量，估算在均匀/balanced gate 下每个 token 平均触达多少通信域。
        # dispatch copy epilogue 的 HBM 读取量约等于 token 数 * 期望 top-k 目标数 * 单 token 字节数。
        # 当前模型不适用于 V3.0 group-limited gate，因为其 expert 选择分布不是全局均匀抽样。
        # TODO: 支持该场景
        assert num_scaleout_topk == 0

        # 自动探测链路带宽：多 RDMA rank 时读取 RDMA 带宽；NVLink 带宽始终用于节点内路径建模。
        if rdma_gbs == 0 and self.num_rdma_ranks > 1:
            rdma_gbs = get_rdma_gbs()
        if nvlink_gbs == 0:
            nvlink_gbs = get_nvlink_gbs()

        # 初始化资源/流量计数。这里估算的是相对单位流量，不直接乘 token 数和 hidden 字节数。
        # 注意：注释中的 “不计 HBM traffic” 指不把 HBM 总量作为瓶颈链路；但仍用 sm_read/sm_write 表示 SM 侧读写压力。
        sm_read, sm_write = 0, 0
        rdma_traffic, nvlink_traffic = 0, 0

        def get_expected_topk(num_groups: int) -> float:
            # 假设 expert 被均匀切分到 `num_groups` 个组中，top-k 从全局 expert 中均匀抽样。
            assert num_experts % num_groups == 0
            # 组合数公式：某个组至少被 top-k 命中的概率 = 1 - 该组所有 expert 都未被选中的概率。
            # 乘以组数后得到一个 token 平均会触达多少个组。
            return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))

        # 一个 token 平均触达的 scale-out 组数；单 scale-out 组时没有跨节点扩展流量。
        num_expected_scaleout_topk = get_expected_topk(self.num_scaleout_ranks) if self.num_scaleout_ranks > 1 else 0

        # 一个 token 平均触达的全局 rank 数，用于归一化每个目标副本承担的读写成本。
        num_expected_topk = get_expected_topk(self.num_ranks)

        # 每个目标副本平均读取一份 token 数据，因此按期望目标数做归一化。
        sm_read += 1 / num_expected_topk

        # 简化模型未考虑“所有 expert 都在本地，跳过 send buffer”的特殊优化路径。
        if self.num_scaleout_ranks > 1:
            # scale-up warp 先把 token 写入发送 buffer，为后续跨节点/节点内转发做准备。
            sm_write += 1 / num_expected_topk

            # scale-out 流量：命中本 scale-out 组可 local bypass，不走 RDMA；其余组产生 RDMA 发送。
            sm_write += (1 / num_expected_topk) * (num_expected_scaleout_topk / self.num_scaleout_ranks)  # 本地旁路
            rdma_traffic += (1 / num_expected_topk) * (num_expected_scaleout_topk * (1 - 1 / self.num_scaleout_ranks))

            # forward warp 从 RDMA 接收/转发数据，并在目标节点内继续通过 NVLink 分发给 scale-up peer。
            sm_read += num_expected_scaleout_topk / num_expected_topk
            sm_write += 1  # 发起 scale-up 通信
            nvlink_traffic += 1 - (1 / self.num_scaleup_ranks)
        else:
            # 非 hybrid scale-out 情况下，如存在多 RDMA rank，先写 send buffer 再由底层 direct 路径发送。
            if self.num_rdma_ranks > 1:
                sm_write += 1 / num_expected_topk

            # 节点内 NVLink 发送占比由物理 NVLink rank 数相对总 rank 数决定。
            sm_write += self.num_nvlink_ranks / self.num_ranks

            # 本地 rank 可 bypass，不产生链路流量；其余节点内走 NVLink，节点外走 RDMA。
            nvlink_traffic += self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)  # 排除本地旁路
            rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks

        # 选择更可能成为瓶颈的链路：比较单位流量除以带宽得到的相对耗时。
        if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
        else:
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

        # 根据瓶颈链路吞吐反推需要多少 SM 才能提供足够的 HBM 读/写供给。
        # 如果不追求与计算重叠，会倾向使用更多 SM 来提高通信 kernel 自身吞吐。
        num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
        num_sms = num_device_sms  # 没有跨 rank 流量时（例如 EP=1）直接使用设备 SM 数作为上限默认值。
        if bounded_traffic > 0:
            num_sms = max(
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
            )
        # 预留 25% 裕量、至少 4 个 SM、并对齐到偶数，匹配底层 kernel/channel 划分习惯。
        num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)
        # 不偏向 overlap 时至少给 64 个 SM，牺牲部分计算并发换通信吞吐。
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
        # 最终不能超过当前 GPU 的物理 SM 数。
        num_sms = min(num_sms, num_device_sms)

        # 调试汇总：打印估算模型中的关键中间量，方便定位 SM 数过大/过小的原因。
        if os.environ.get('EP_BUFFER_DEBUG', 0):
            print(f'EP SM approximation: '
                  f'{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, '
                  f'{rdma_gbs=}, {nvlink_gbs=}, '
                  f'{num_expected_scaleout_topk=}, {num_expected_topk=}, '
                  f'{bounded_traffic=}, {bounded_gbs=}, {num_sms=}')
        return num_sms

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        """
        根据 SM 数和通信模式估算最优 RDMA QP 数。

        第一性原理说明：
            QP 是 RDMA 的并发通道抽象。数据被拆成多个 channel 后，如果所有 channel 共享过少 QP，
            会在网卡队列上排队；如果每个细粒度任务都独占 QP，又会增加资源和提交开销。因此这里根据
            direct/hybrid 模式选择不同策略：direct 偏少 QP 降低开销，hybrid 偏多 QP 避免多层转发互相阻塞。

        参数：
            num_sms：dispatch/combine kernel 使用的 SM 数。

        返回：
            num_qps：推荐的 QP 数，并受 `num_allocated_qps` 上限约束。
        """
        # direct 模式下 QP 太多会增加 doorbell ringing 等提交开销，因此限制在最多 8 个数据 QP + 1 个 notify QP。
        num_qps = min(num_sms, 8 + 1)

        # hybrid 模式中每个 SM/channel 组合倾向使用独立 QP，减少多 channel 竞争；额外 +1 给 notify。
        if self.allow_hybrid_mode:
            num_qps = num_sms * 16 + 1

        # 不能超过构造时预分配的 QP 上限，否则底层 RDMA 资源不存在。
        return min(num_qps, self.num_allocated_qps)

    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 num_experts: Optional[int] = None,
                 num_max_tokens_per_rank: Optional[int] = None,
                 expert_alignment: Optional[int] = None,
                 num_sms: int = 0, num_qps: int = 0,
                 previous_event: Optional[EventHandle] = None,
                 previous_event_before_epilogue: Optional[EventHandle] = None,
                 async_with_compute_stream: bool = False,
                 allocate_on_comm_stream: bool = False,
                 handle: Optional[EPHandle] = None,
                 do_handle_copy: bool = True,
                 do_cpu_sync: Optional[bool] = None,
                 do_expand: bool = False,
                 use_tma_aligned_col_major_sf: bool = False) \
            -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     Optional[torch.Tensor], Optional[torch.Tensor],
                     EPHandle, EventOverlap]:
        """
        将 token 分发到不同 rank，支持单机和多机场景。
        如果未显式指定，SM 数和 QP 数会自动决定。

        第一性原理说明：
            dispatch 解决的是 MoE 中的“scatter by key”问题：每个 token 的 key 是 expert id，expert id 决定目标 rank。
            由于 top-k 可能让一个 token 发送到多个 expert，dispatch 既要搬运 token 数据，也要生成足够的 metadata，
            让接收端知道哪些 token 属于哪个 local expert，让 combine 端知道如何回到原始 token 顺序。

            这个函数的参数可以分为四类：
            1. 数据本体：`x`、`sf`、`topk_weights`；
            2. 路由描述：`topk_idx`、`num_experts`、`expert_alignment`、`do_expand`；
            3. 资源调度：`num_sms`、`num_qps`、stream/event 相关参数；
            4. 复用与同步：`handle`、`do_handle_copy`、`do_cpu_sync`。
            这种分层有助于理解为什么 dispatch 不只是一个 tensor 输入输出函数，而是一次完整的通信计划生成与执行。

        参数：
            x：`torch.Tensor` 或 `torch.Tensor` tuple。第一种形式要求形状为 `[num_tokens, hidden]`，
                类型为 `torch.bfloat16`；第二种形式用于 FP8 模式，tuple 第一个元素形状为
                `[num_tokens, hidden]`、类型为 `torch.float8_e4m3fn`，第二个元素为 scale factor。
            topk_idx：每个 token 选择的 expert 索引，形状为 `[num_tokens, num_topk]`，类型为
                `deep_ep.topk_idx_t`（通常为 `torch.int64`）；`-1` 表示没有选择。
                如果提供了 `handle`，则该参数必须为 `None`。
            topk_weights：每个 token 要 dispatch 的 expert 权重，形状为 `[num_tokens, num_topk]`，
                类型为 `torch.float`。如果提供了 `handle`，则该参数必须为 `None`。
            cumulative_local_expert_recv_stats：累计 expert 计数统计 tensor，形状为 `[num_local_experts]`，
                类型为 `torch.int`，可用于在线 EP 负载均衡监控。
            num_experts：全局 expert 总数；如果提供了 `handle`，可从其中推断。
            num_max_tokens_per_rank：每个 rank 的最大 token 数；可从构造函数默认值或 `handle` 推断。
            expert_alignment：每个 local expert 接收 token 数的对齐粒度。
            num_sms：使用的 SM 数，0 表示通过 `get_theoretical_num_sms` 自动决定。
            num_qps：使用的 RDMA QP 数，0 表示通过 `get_theoretical_num_qps` 自动决定。
            previous_event：实际执行 kernel 前需要等待的 event。如果设置该参数，`allocate_on_comm_stream`
                也必须为 `True`。
            previous_event_before_epilogue：实际执行 copy epilogue 前需要等待的 event。
            async_with_compute_stream：如果设置，当前 stream 不会等待通信 kernel 完成。
            allocate_on_comm_stream：控制所有已分配 tensor 的所有权是否归属通信 stream。
            handle：可选的上一轮 dispatch 返回的缓存 `EPHandle`。如果设置，CPU 会复用布局信息以节省时间；
                此时 `topk_idx` 和 `topk_weights` 必须为 `None`。
            do_handle_copy：返回 handle 时是否 clone `topk_idx`，用于防止用户修改原始输入。
            do_cpu_sync：是否与 CPU 同步以获取精确接收 token 数；除非提供了 `handle`，`None` 默认等价于 `True`。
            do_expand：是否使用展开布局，即每个 token-expert 选择独占一个 slot。
            use_tma_aligned_col_major_sf：scale factor 是否使用 TMA 对齐的列主序布局。

        返回：
            recv_x：接收到的 token，类型和 tuple 结构与输入 `x` 一致。
            recv_topk_idx：接收到的 expert 索引。
            recv_topk_weights：接收到的 expert 权重；如果未提供 `topk_weights`，则为 `None`。
            handle：返回的通信句柄。
            event：kernel 执行后的 event，仅在设置 `async_with_compute_stream` 时有效。
        """
        # DeepEP 的某些高性能路径与 PyTorch deterministic 算法设置不兼容，因此入口处统一检查。
        check_torch_deterministic()

        # 自动决定通信 kernel 使用的 SM 数和 RDMA QP 数。
        # 若复用 handle，则 top-k 维度来自 handle；否则来自当前传入的 `topk_idx`。
        num_topk = (handle.topk_idx if topk_idx is None else topk_idx).shape[1]
        # num_sms=0 表示按带宽模型估算；显式传入时尊重调用方设置。
        num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else num_sms
        # num_qps=0 表示根据 SM 数和模式自动推导 QP 数。
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
        # 运行期申请的 QP 不能超过构造时预分配数量。
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'

        # FP8 dispatch 输入是 `(x, sf)`：x 为 FP8 token，sf 为 scale factor；BF16 输入没有 sf。
        # 第一性原理：低精度数据只保存近似值，必须携带 scale factor 才能表达原始动态范围；因此数据面被拆成值和尺度两部分。
        x, sf = x if isinstance(x, tuple) else (x, None)

        # 拆解并校验缓存 handle。复用 handle 时，路由布局固定，不能再传新的 topk_idx/topk_weights。
        # 第一性原理：缓存布局等价于承诺“本次 token-to-slot 映射与上次相同”。如果路由变了还复用旧 handle，
        # 数据会被写到错误 expert/rank 的槽位，combine 也会按旧路径规约，结果在数值上不可解释。
        if handle is not None:
            # cached dispatch 只复用旧路由，因此禁止同时传入新的路由索引和权重。
            assert topk_idx is None and topk_weights is None
            # cached handle 的 token 计数来自上次结果，不能再要求 CPU 同步重新取精确计数。
            assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'
            # 路由索引从 handle 恢复，确保本次 dispatch 与上次布局完全一致。
            topk_idx = handle.topk_idx
            # 如果调用方没有传入这些参数，则继承 handle 中记录的上下文。
            num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
            num_experts = value_or(num_experts, handle.num_experts)
            expert_alignment = value_or(expert_alignment, handle.expert_alignment)
            # 复用 handle 时默认走纯 GPU/cache 路径，避免 CPU 同步破坏性能。
            do_cpu_sync = False

            # handle 的关键上下文必须与本次参数一致，否则 cached slot/expert offset 会错位。
            assert (num_experts, expert_alignment, num_max_tokens_per_rank) == \
                   (handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank)
        # 将 handle 中可复用的布局元数据拆成 runtime.dispatch 的 cached 参数。
        (cached_num_recv_tokens, cached_num_recv_tokens_per_expert_list,
         cached_psum_num_recv_tokens_per_scaleup_rank, cached_psum_num_recv_tokens_per_expert,
         cached_dst_buffer_slot_idx,
         cached_token_metadata_at_forward,
         cached_channel_linked_list) = self._unpack_handle(handle)

        # 默认值收敛：调用方未指定时使用构造默认值或最保守设置。
        # 第一性原理：分布式 API 的每个 rank 必须对容量和对齐做出相同决策；集中在这里补齐默认值，
        # 可以减少调用方在不同 rank 上因 None/默认参数处理不一致而造成的隐性错配。
        num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)
        expert_alignment = value_or(expert_alignment, 1)
        do_cpu_sync = value_or(do_cpu_sync, True)

        # 调用底层 C++/CUDA 扩展执行 dispatch。返回值包括接收张量、路由元数据、缓存句柄所需信息和 CUDA event。
        (recv_x, recv_sf,
         recv_topk_idx, recv_topk_weights,
         cloned_topk_idx,
         num_recv_tokens_per_expert_list,
         psum_num_recv_tokens_per_scaleup_rank,
         psum_num_recv_tokens_per_expert,
         recv_src_metadata,
         dst_buffer_slot_idx,
         token_metadata_at_forward,
         channel_linked_list,
         event) = self.runtime.dispatch(x, sf, topk_idx, topk_weights,
                                        cumulative_local_expert_recv_stats,
                                        cached_num_recv_tokens,
                                        cached_num_recv_tokens_per_expert_list,
                                        cached_psum_num_recv_tokens_per_scaleup_rank,
                                        cached_psum_num_recv_tokens_per_expert,
                                        cached_dst_buffer_slot_idx,
                                        cached_token_metadata_at_forward,
                                        cached_channel_linked_list,
                                        num_max_tokens_per_rank,
                                        num_experts, expert_alignment,
                                        num_sms, num_qps,
                                        previous_event,
                                        previous_event_before_epilogue,
                                        async_with_compute_stream, allocate_on_comm_stream,
                                        do_handle_copy, do_cpu_sync, do_expand,
                                        use_tma_aligned_col_major_sf)
        # 首次 dispatch 需要构造新的 EPHandle；复用 handle 时继续返回原 handle。
        if handle is None:
            handle = EPHandle(do_expand,
                              num_experts, expert_alignment,
                              num_max_tokens_per_rank,
                              num_sms,
                              # 默认使用底层返回的 cloned_topk_idx，防止用户原地修改输入 topk_idx 破坏 combine。
                              cloned_topk_idx if do_handle_copy else topk_idx,
                              num_recv_tokens_per_expert_list,
                              psum_num_recv_tokens_per_scaleup_rank,
                              psum_num_recv_tokens_per_expert,
                              recv_src_metadata,
                              dst_buffer_slot_idx,
                              token_metadata_at_forward,
                              channel_linked_list)

        # FP8 路径需要把接收到的 token 和 scale factor 重新打包成 `(recv_x, recv_sf)`。
        recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x

        # EventOverlap 封装底层 event；async 模式下调用方可据此等待通信完成或与计算重叠。
        return recv_x, recv_topk_idx, recv_topk_weights, handle, EventOverlap(event)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        规范化 combine epilogue 的 bias 输入。

        第一性原理说明：
            combine epilogue 已经会读取并写回最终输出，如果 bias 也要参与最终结果，把 bias 加法融合到 epilogue
            可以少启动一次 kernel、少读写一次 HBM。Python 允许灵活的 Union 输入，但底层 kernel 需要固定参数布局，
            因此这里把“灵活 API”转换成“固定 ABI”。

        combine 支持无 bias、一个 bias tensor，或两个 bias tensor。Python 层统一拆成 `(bias_0, bias_1)`，
        让 C++ runtime 不需要处理 Python 的 Union 类型。
        """
        # 默认无 bias。
        bias_0, bias_1 = None, None
        # 单 tensor bias：只填第一个槽位。
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        # 双 bias：常用于把两个额外项融合进 combine epilogue，减少单独 kernel launch/访存。
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    def combine(self,
                x: torch.Tensor,
                handle: EPHandle,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                num_sms: int = 0, num_qps: int = 0,
                previous_event: EventHandle = None,
                previous_event_before_epilogue: Optional[EventHandle] = None,
                async_with_compute_stream: bool = False,
                allocate_on_comm_stream: bool = False) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        将来自不同 rank 的 token combine（规约）回它们的原始 rank。
        支持单机和多机场景。

        第一性原理说明：
            combine 是 dispatch 的逆过程，但不是简单“按原路发回”。一个 token 可能被多个 expert 处理，
            返回时需要把多个 expert 输出按 top-k 权重加权求和，并写回原始 token 的位置。因此 combine 同时包含
            数据回传、按来源 metadata 定位、以及 reduce epilogue 三件事。

            这里必须传入 dispatch 返回的 `handle`，因为只有它记录了正向路由的可逆映射。如果没有 handle，
            combine 无法从 expert 输出 tensor 的排列推断每一行应该回到哪个原始 rank/token，也不知道 hybrid 模式
            下哪些 token 经过了中间转发路径。

        参数：
            x：待发送并规约回原始 rank 的 token，形状为 `[num_tokens, hidden]`，类型为 `torch.bfloat16`。
            handle：必须设置的通信句柄，可从 `dispatch` 函数获取。
            topk_weights：用于规约回原始 rank 的 token top-k 权重，形状为 `[num_tokens, num_topk]`，
                类型为 `torch.float`；expand 模式下不使用。
            bias：0、1 或 2 个最终加到输出上的 bias，形状为 `[num_combined_tokens, hidden]`，
                类型为 `torch.bfloat16`。
            num_sms：使用的 SM 数，0 表示复用 dispatch handle 中记录的 SM 数。
            num_qps：使用的 RDMA QP 数，0 表示通过 `get_theoretical_num_qps` 自动决定。
            previous_event：实际执行 kernel 前需要等待的 event。如果设置该参数，`allocate_on_comm_stream`
                也必须为 `True`。
            previous_event_before_epilogue：实际执行 reduce epilogue 前需要等待的 event。
            async_with_compute_stream：如果设置，当前 stream 不会等待通信 kernel 完成。
            allocate_on_comm_stream：控制所有已分配 tensor 的所有权是否归属通信 stream。

        返回：
            combined_x：规约后的 token tensor，形状为 `[num_combined_tokens, hidden]`，类型为 `torch.bfloat16`。
            combined_topk_weights：规约后的 top-k 权重，形状为 `[num_combined_tokens, num_topk]`，类型为 `torch.float`。
            event：kernel 执行后的 event，仅在设置 `async_with_compute_stream` 时有效。
        """
        # 与 dispatch 一样，combine 入口处检查 PyTorch deterministic 配置，避免启用不兼容路径。
        check_torch_deterministic()

        # combine 默认复用 dispatch 记录的 SM 数，保持正反向通信 kernel 的资源模型一致。
        num_sms = handle.num_sms if num_sms == 0 else num_sms
        # QP 数仍根据当前 SM 数自动推导；调用方也可显式传入用于调优。
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
        # 运行期 QP 数必须落在构造时预分配的资源范围内。
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'

        # 将 Python 侧 bias Union 拆成底层固定的两个可选 tensor 参数。
        bias_0, bias_1 = ElasticBuffer._unpack_bias(bias)
        # 调用底层 combine：依据 dispatch 保存的来源元数据和 top-k 路由，把 expert 输出规约回原始 token 所在 rank。
        # 第一性原理：规约的正确性依赖“相同 token 的多个 expert 输出最终落到同一 accumulator”。
        # `recv_src_metadata` 给出 accumulator 的地址，`topk_idx`/`topk_weights` 给出参与规约的 expert 维度语义。
        combined_x, combined_topk_weights, event = \
            self.runtime.combine(x, topk_weights,
                                 bias_0, bias_1,
                                 handle.recv_src_metadata,
                                 handle.topk_idx,
                                 handle.psum_num_recv_tokens_per_scaleup_rank,
                                 handle.token_metadata_at_forward,
                                 handle.channel_linked_list,
                                 handle.num_experts,
                                 handle.num_max_tokens_per_rank,
                                 num_sms, num_qps,
                                 previous_event,
                                 previous_event_before_epilogue,
                                 async_with_compute_stream,
                                 allocate_on_comm_stream,
                                 handle.do_expand)
        # 返回规约后的 token、可选 top-k weight，以及封装后的 CUDA event。
        return combined_x, combined_topk_weights, EventOverlap(event)
