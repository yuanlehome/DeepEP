# test_ep.py 日志字段解读

## 1. 头部配置

```
Config:
 > Ranks: 1 x 8          # scaleout_ranks(跨节点) x scaleup_ranks(节点内 GPU 数)
 > Experts: 6/256         # top-k / 总专家数
 > Tokens: 4096 (max: 4096), hidden: 7168
 > #SM: 64, #QPs: 129/129 # 使用的 SM 数 / 实际使用 QP 数 / 分配 QP 数
```

- **Ranks**: `scaleout × scaleup`，例如 `1 x 8` 表示单节点 8 卡，`4 x 8` 表示 4 节点每节点 8 卡
- **Experts**: `num_topk / num_experts`，每个 token 选 top-k 个专家，共 num_experts 个专家
- **#SM**: 通信 kernel 占用的 SM 数，由 `--num-sms` 控制（0 为自动推导）
- **#QPs**: `实际使用 QP 数 / 分配 QP 数`，QP (Queue Pair) 是 RDMA 通信资源

---

## 2. 每条性能行格式

### 前缀符号说明

| 符号 | 操作类型 | 说明 |
|------|----------|------|
| `*`  | dispatch | 标准分发，含完整路由+数据传输 |
| `-`  | expanded dispatch | 按 expert 展开布局（`do_expand=True`），便于 TMA 访问 |
| `#`  | cached dispatch | 复用上次路由 handle，只重传 x 数据 |
| `@`  | combine | 标准 combine，各 rank 聚合来自 expert 的输出 |
| `+`  | reduced combine | expand 模式下的 combine，含 intra-rank reduce 步骤 |

### 字段格式

```
<符号> EP: <rank>/<total> | <操作>: <SO带宽> GB/s (SO), <SU带宽> GB/s (SU), <总耗时> us, <传输字节数> bytes | <后处理类型>: <后处理带宽> GB/s, <后处理耗时> us
```

**示例：**
```
* EP:   3/8 | dispatch: 0 GB/s (SO), 703 GB/s (SU), 193.153 us, 135747768 bytes | copy: 6492 GB/s, 41.823 us
@ EP:   7/8 | combine:  0 GB/s (SO), 719 GB/s (SU), 362.901 us, 260806320 bytes | reduce: 1922 GB/s, 61.098 us
```

### 各字段含义

| 字段 | 含义 |
|------|------|
| `rank/total` | 当前 rank 序号 / 总 rank 数 |
| `GB/s (SO)` | **Scale-Out 带宽**：跨节点 IB/RoCE 网卡流量；单节点时恒为 0 |
| `GB/s (SU)` | **Scale-Up 带宽**：节点内 NVLink 流量，**核心性能指标** |
| `us` | **总通信耗时**，含 barrier 同步等待时间（`bench_kineto` 测量） |
| `bytes` | 该 rank 通过 scaleup 通道实际传输的数据量 |
| `copy: GB/s, us` | dispatch 后处理（`dispatch_copy_epilogue_impl`）的显存带宽和耗时 |
| `reduce: GB/s, us` | combine 后处理（`combine_reduce_epilogue_impl`）的显存带宽和耗时，包含 bias + 读 + 写 |

---

## 3. 带宽计算公式

各操作的 SU/SO 带宽计算逻辑（来自 `test_ep.py`）：

**dispatch (`*` / `-` / `#`)：**
```
SU 带宽 = (每token字节数 × scaleup接收token数) / 总耗时
SO 带宽 = (每token字节数 × scaleout发送token数) / 总耗时
copy 带宽 = 2 × scaleup接收字节数 / copy耗时   # 读+写
```

**combine (`@` / `+`)：**
```
SU 带宽 = (每token字节数 × scaleup发送token数) / 总耗时
reduce 带宽 = (bias字节数 + reduce读字节数 + reduce写字节数) / reduce耗时
```

---

## 4. 参考性能区间（单节点 8 卡）

| 指标 | H100 典型范围 | B300 参考范围 | 说明 |
|------|---------------|---------------|------|
| NVLink SU 带宽 | 600–800 GB/s | 700–720 GB/s | H100 NVLink 4 单向峰值 ~450 GB/s；B30Z NVLink 5 单向峰值 ~900 GB/s，P2P 实测约 715 GB/s |
| HBM reduce 带宽 | 1500–2500 GB/s | 1900–2100 GB/s | B30Z HBM3e 峰值 ~8 TB/s，reduce 受 bias+读+写 pattern 限制，实测见下 |
| dispatch 耗时 | 180–220 us | 190–200 us | B30Z 单节点实测，NVLink 5 transfer 耗时与 H100 相近 |
| combine 耗时 | 350–400 us | 360–380 us | B30Z 单节点实测 |

**说明**：B30Z（即 B300 国内版）NVLink 5 单流 P2P 带宽实测约 715 GB/s，与 test_ep.log 中 SU 700–728 GB/s 吻合，说明 DeepEP 已充分利用 NVLink 5；HBM reduce 带宽偏低（~2000 GB/s）是 combine reduce epilogue 的 bias+读+写 访问模式导致的有效利用率问题，非硬件瓶颈。NCCL AllReduce bus_bw 实测 571 GB/s（0.25 GB tensor，8 GPU ring），低于理论峰值 1620 GB/s，主要原因是 0.25 GB 小 tensor 受启动开销影响大，可用更大 tensor（如 1 GB）复测。

---

## 5. 典型日志样例解读

```
 > Testing with do_handle_copy=1, expert_alignment=128, use_fp8_dispatch=1, num_bias=0,
   with_previous_event=0, async_with_compute_stream=0, allocate_on_comm_stream=0 ...
```

本轮测试参数含义：

| 参数 | 值 | 含义 |
|------|----|------|
| `do_handle_copy` | 1 | 对 topk_idx 做深拷贝，handle 与原张量独立 |
| `expert_alignment` | 128 | 每个 expert 接收的 token 数按 128 对齐（padding）|
| `use_fp8_dispatch` | 1 | dispatch 的 x 数据使用 FP8 量化传输 |
| `num_bias` | 0 | combine 时不叠加 bias |
| `with_previous_event` | 0 | 不依赖上一个 CUDA event（不做 event 链式等待）|
| `async_with_compute_stream` | 0 | 不与 compute stream 异步重叠 |
| `allocate_on_comm_stream` | 0 | buffer 分配在默认流，不在 comm stream |

---

## 6. SO 带宽为 0 的情况

日志中所有 `SO = 0 GB/s` 是正常现象，原因：

- 本次配置 `Ranks: 1 x 8`（单节点），不存在跨节点流量
- `num_scaleout_ranks = 1`，代码中 `if num_scaleout_ranks == 1: num_scaleout_tokens = 0`

多节点场景（如 `4 x 8`）下 SO 带宽才会有非零值，反映 IB/RoCE 网卡性能。
