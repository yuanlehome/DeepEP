# Day 01：DeepEP 仓库地图

这份地图按五层阅读 DeepEP：**构建 → Python API → C++ runtime → kernel → tests**。整体调用链是：测试脚本调用 `deep_ep` Python 包，Python 包加载 `deep_ep._C` 扩展，`_C` 通过 pybind11 进入 C++ runtime，runtime 再调度静态编译或 JIT 编译的 CUDA kernel。

```text
构建层
  setup.py / CMakeLists.txt / develop.sh
        │ 生成 deep_ep._C 扩展、envs.py、链接 CUDA/NCCL/NVSHMEM
        ▼
Python API 层
  deep_ep/__init__.py
  deep_ep/buffers/elastic.py
  deep_ep/buffers/legacy.py
        │ 封装 _C.ElasticBuffer / _C.Buffer，提供 dispatch/combine 等接口
        ▼
C++ runtime 层
  csrc/python_api.cpp
  csrc/elastic/buffer.hpp
  csrc/legacy/buffer.hpp
  csrc/jit/*.hpp
        │ 管理通信资源、buffer、barrier、JIT 编译和 pybind 注册
        ▼
Kernel 层
  csrc/kernels/elastic/*.hpp
  csrc/kernels/legacy/*.cu
  csrc/kernels/backend/*.cu
  deep_ep/include/deep_ep/impls/*.cuh
        │ 执行 EP 通信、combine、barrier、Engram、PP、AGRS 等 GPU 工作
        ▼
Tests 层
  tests/elastic/*.py
  tests/legacy/*.py
  tests/utils/*.py
```

## 1. 构建层

构建层负责把 Python 包、C++ 绑定、CUDA kernel、外部通信库打包成可 import 的 `deep_ep._C` 扩展。

- `setup.py` 是主构建入口。它查找 NVSHMEM/NCCL，设置编译和链接参数，并通过 `CUDAExtension(name='deep_ep._C', ...)` 生成扩展；源码列表包含 `csrc/python_api.cpp`、legacy CUDA kernel、backend CUDA driver/NCCL/NVSHMEM 实现。
- `setup.py` 的 `CustomBuildPy` 会把构建时的持久化环境变量写入 `deep_ep/envs.py`，例如 `EP_JIT_CACHE_DIR`、`EP_NCCL_ROOT_DIR`。
- `CMakeLists.txt` 标注为 debug 用途，主要给 CMake/IDE 调试和索引用；正式安装路径仍以 Torch extension 构建为主。
- `develop.sh` 是本地开发脚本：设置 NVSHMEM、CUDA arch、JIT cache，运行 `python setup.py build`，然后把生成的 `.so` 链接回 `deep_ep/`，并复制 `envs.py`。

关键文件：

- `setup.py:49`：自定义 build_py，生成默认环境变量文件。
- `setup.py:79`：声明 C++/CUDA 源文件列表。
- `setup.py:180`：定义 `deep_ep._C` CUDA 扩展。
- `CMakeLists.txt:1`：说明 CMake 仅用于 debug。
- `develop.sh:14`：执行开发构建。

## 2. Python API 层

Python API 层是用户直接 import 和调用的入口，负责初始化环境、检查依赖、封装 C++ runtime 对象。

- `deep_ep/__init__.py` 在 import 时先加载 `deep_ep/envs.py` 中的持久化环境变量，再检查运行时 NCCL 是否与链接的 NCCL 一致。
- `deep_ep/__init__.py` 调用 `_C.init_jit(library_root_path, cuda_home, nccl_root)` 初始化 JIT 编译运行时。
- `deep_ep/buffers/elastic.py` 定义 V2 的 `ElasticBuffer` 和 `EPHandle`，支持 high-throughput EP all-to-all、Engram、PP send/recv、AGRS 等能力。
- `deep_ep/buffers/legacy.py` 定义 legacy `Buffer`，支持 intranode、internode、low-latency 三类 MoE all-to-all 通信路径。
- `deep_ep/utils/` 放置 Python 侧辅助逻辑，包括通信句柄、环境检查、测试参考实现、gate、事件封装等。

关键文件：

- `deep_ep/__init__.py:10`：加载构建期持久化环境变量。
- `deep_ep/__init__.py:46`：检查 NCCL so 是否冲突或版本不一致。
- `deep_ep/__init__.py:71`：初始化 JIT runtime。
- `deep_ep/__init__.py:88`：导出 `Buffer`、`ElasticBuffer`、`EPHandle`。
- `deep_ep/buffers/elastic.py:90`：定义 `ElasticBuffer`。
- `deep_ep/buffers/elastic.py:205`：创建 `_C.ElasticBuffer` C++ runtime。
- `deep_ep/buffers/legacy.py:14`：定义 legacy `Buffer`。
- `deep_ep/buffers/legacy.py:92`：创建 `_C.Buffer` C++ runtime。

## 3. C++ runtime 层

C++ runtime 层是 Python 与 CUDA kernel 之间的调度中心，负责 pybind11 绑定、通信资源管理、buffer 生命周期、JIT 编译入口和 kernel launch 参数组织。

- `csrc/python_api.cpp` 是 `deep_ep._C` 的 pybind11 模块入口。它注册 JIT API、legacy buffer API、elastic buffer API。
- `csrc/jit/api.hpp` 暴露 `init_jit` 绑定，接收 Python 传入的 library root、CUDA root、NCCL root。
- `csrc/jit/compiler.hpp`、`kernel_runtime.hpp`、`launch_runtime.hpp` 等文件组成 JIT 基础设施，负责 include 解析、缓存、编译 cubin、加载和 launch。
- `csrc/elastic/buffer.hpp` 是 V2 runtime 主体，管理 NCCL 对称内存、GPU/CPU workspace、barrier、dispatch/combine、Engram、PP、AGRS 等路径。
- `csrc/legacy/buffer.hpp` 是 legacy runtime 主体，管理 NVLink/NVSHMEM/RDMA 相关 buffer、IPC/NVSHMEM 同步和 low-latency 路径。
- `csrc/utils/` 放置 C++ 侧通用工具，例如 lazy driver、shared memory、event、format、system 等。

关键文件：

- `csrc/python_api.cpp:22`：定义 pybind11 模块。
- `csrc/python_api.cpp:32`：注册 JIT API。
- `csrc/python_api.cpp:35`：注册 legacy buffer API。
- `csrc/python_api.cpp:38`：注册 elastic buffer API。
- `csrc/jit/api.hpp:9`：初始化 JIT 子系统。
- `csrc/elastic/buffer.hpp`：ElasticBuffer C++ runtime。
- `csrc/legacy/buffer.hpp`：Legacy Buffer C++ runtime。

## 4. Kernel 层

Kernel 层是真正执行 GPU 通信和数据搬运/归约的地方。仓库里同时存在 V2 elastic JIT kernel 和 legacy 静态 CUDA kernel。

- `csrc/kernels/elastic/api.hpp` 汇总 V2 kernel 入口，包含 barrier、dispatch、combine、engram、PP send/recv。
- `csrc/kernels/elastic/dispatch.hpp`、`combine.hpp`、`engram.hpp` 等文件组织 JIT runtime，按运行时形状和配置生成/加载具体 kernel。
- `deep_ep/include/deep_ep/impls/` 存放 JIT 编译时会被包含的实现模板，是 elastic kernel 的主要实现来源。
- `csrc/kernels/legacy/` 存放 legacy 静态编译 CUDA kernel，包括 layout、intranode、internode、internode low-latency 等路径。
- `csrc/kernels/backend/` 封装底层后端能力，如 CUDA driver、NCCL、NVSHMEM。
- `csrc/indexing/main.cu` 主要服务 CMake-based IDE kernel 代码索引。

关键文件：

- `csrc/kernels/elastic/api.hpp:5`：引入 barrier kernel。
- `csrc/kernels/elastic/api.hpp:6`：引入 dispatch kernel。
- `csrc/kernels/elastic/api.hpp:7`：引入 combine kernel。
- `csrc/kernels/elastic/api.hpp:8`：引入 engram kernel。
- `csrc/kernels/elastic/api.hpp:9`：引入 PP send/recv kernel。
- `setup.py:88`：把 legacy internode、low-latency、NVSHMEM backend 加入静态编译。
- `setup.py:95`：把 NCCL backend 加入编译。
- `setup.py:100`：把 CUDA driver backend 加入编译。

## 5. Tests 层

Tests 层从 Python API 侧验证功能正确性、通信路径和性能指标，覆盖 elastic V2、legacy、utils 三类。

- `tests/elastic/test_ep.py` 测试 ElasticBuffer 的 EP dispatch/combine 路径，并与 Python 参考实现做正确性对比。
- `tests/elastic/test_pp.py` 测试 pipeline-parallel send/recv。
- `tests/elastic/test_engram.py` 测试 Engram write/fetch。
- `tests/elastic/test_barrier.py` 测试 elastic runtime barrier。
- `tests/elastic/test_agrs.py` 测试 all-gather reduce-scatter。
- `tests/legacy/test_intranode.py` 测试 legacy NVLink intranode 路径。
- `tests/legacy/test_internode.py` 测试 legacy RDMA/internode 路径。
- `tests/legacy/test_low_latency.py` 测试 legacy low-latency 路径。
- `tests/utils/test_gate.py` 测试 Python utils 中 gate 相关逻辑。

关键文件：

- `tests/elastic/test_ep.py`：Elastic EP 主测试。
- `tests/elastic/test_pp.py`：Elastic PP 测试。
- `tests/elastic/test_engram.py`：Elastic Engram 测试。
- `tests/elastic/test_barrier.py`：Elastic barrier 测试。
- `tests/elastic/test_agrs.py`：Elastic AGRS 测试。
- `tests/legacy/test_intranode.py`：Legacy intranode 测试。
- `tests/legacy/test_internode.py`：Legacy internode 测试。
- `tests/legacy/test_low_latency.py`：Legacy low-latency 测试。

## 两条主线

### Elastic / DeepEP V2 主线

```text
tests/elastic/*.py
  → deep_ep.ElasticBuffer
  → _C.ElasticBuffer
  → csrc/elastic/buffer.hpp
  → csrc/kernels/elastic/*.hpp
  → deep_ep/include/deep_ep/impls/*.cuh
```

这条线偏向新 runtime：Python 侧提供统一 buffer API，C++ 侧管理 NCCL 对称内存和通信资源，kernel 多通过 JIT 根据配置生成。

### Legacy 主线

```text
tests/legacy/*.py
  → deep_ep.Buffer
  → _C.Buffer
  → csrc/legacy/buffer.hpp
  → csrc/kernels/legacy/*.cu
  → csrc/kernels/backend/*.cu
```

这条线偏向旧 runtime：Python 侧设置 NVSHMEM/NVLink/RDMA 环境，C++ runtime 管理 IPC/NVSHMEM 同步，kernel 主要在构建阶段静态编译进 `_C`。

## 阅读建议

1. 先看 `setup.py` 和 `deep_ep/__init__.py`，弄清楚 `_C` 如何生成、如何初始化。
2. 再看 `deep_ep/buffers/elastic.py` 和 `deep_ep/buffers/legacy.py`，从用户 API 角度理解参数和返回值。
3. 顺着 `csrc/python_api.cpp` 找到 C++ 注册入口，再进入 `csrc/elastic/buffer.hpp` 或 `csrc/legacy/buffer.hpp`。
4. 对 V2 路径继续看 `csrc/kernels/elastic/*.hpp` 与 `deep_ep/include/deep_ep/impls/`；对 legacy 路径看 `csrc/kernels/legacy/*.cu`。
5. 最后用 `tests/elastic/` 和 `tests/legacy/` 反向验证每条调用链。
