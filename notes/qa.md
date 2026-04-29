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
