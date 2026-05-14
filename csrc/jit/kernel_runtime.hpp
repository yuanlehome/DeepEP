#pragma once

// PyTorch CUDA 上下文接口。当前文件不直接调用 CUDAContext，通常由 JIT 运行时模块整体依赖引入。
#include <ATen/cuda/CUDAContext.h>

// DeepEP 的 host/device 断言宏，加载 cubin 或解析符号失败时用于快速报错。
#include <deep_ep/common/exception.cuh>

// 格式化工具，用于拼接 cuobjdump 命令和错误信息。
#include "../utils/format.hpp"
// 懒初始化工具。当前文件不直接使用，但与 JIT runtime/cache 模块的初始化机制保持一致。
#include "../utils/lazy_init.hpp"
// CUDA driver handle 封装：LibraryHandle、KernelHandle、load_kernel、unload_library 等定义在这里。
#include "handle.hpp"

// JIT 运行时相关逻辑放在 deep_ep::jit 命名空间下。
namespace deep_ep::jit {

// KernelRuntime 表示一个已经编译并可执行的 JIT kernel 运行时对象。
// 它负责从缓存目录加载 kernel.cubin，解析唯一 kernel 符号，并持有 CUDA library/kernel handle。
class KernelRuntime final {
public:
    // CUDA Toolkit 根目录，由 Python 侧初始化传入，用于定位 cuobjdump。
    static std::filesystem::path cuda_home;

    // CUDA library/module 句柄，代表已加载的 cubin 模块。
    LibraryHandle library;
    // CUDA kernel/function 句柄，代表 cubin 中真正可 launch 的入口函数。
    KernelHandle kernel;

    // 从一个 JIT 缓存目录构造运行时对象；目录中应至少包含 kernel.cu 和 kernel.cubin。
    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        // `prepare_init` 必须已经设置 cuda_home，否则无法定位 cuobjdump。
        EP_HOST_ASSERT(not cuda_home.empty());

        // NOLINT(*-pro-type-member-init)
        // cuobjdump 与 nvcc 使用同一个 CUDA Toolkit 目录，避免不同 CUDA 版本工具混用。
        const auto cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        // JIT 编译器固定把二进制产物命名为 kernel.cubin。
        const auto cubin_path = dir_path / "kernel.cubin";
        // Debug 模式下打印当前加载的 cubin 路径，便于定位缓存命中和文件来源。
        if (get_env<int>("EP_JIT_DEBUG"))
            printf("Loading CUBIN: %s\n", cubin_path.c_str());

        // Find the only symbol
        // TODO: use kernel enumeration for newer drivers
        // cuobjdump -symbols 会列出 cubin 中的函数符号；这里通过文本过滤找出唯一 kernel 入口。
        // 这些名字是 cubin 中可能出现但不应当作为用户 kernel 入口的内部/辅助符号。
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};
        // 调用 cuobjdump 提取符号表输出，后续逐行解析。
        const auto [exit_code, symbols] = call_external_command(fmt::format("{} -symbols {}", cuobjdump_path.c_str(), cubin_path.c_str()));
        // cuobjdump 必须成功，否则说明 cubin 不存在、损坏，或 CUDA 工具链不可用。
        EP_HOST_ASSERT(exit_code == 0);
        // 把符号表字符串包装成输入流，方便按行扫描。
        std::istringstream iss(symbols);
        // 收集符合条件的 kernel 入口符号名。
        std::vector<std::string> symbol_names;
        // 逐行遍历 cuobjdump 输出。
        for (std::string line; std::getline(iss, line); ) {
            // `STT_FUNC` 表示函数符号，`STO_ENTRY` 表示 CUDA kernel entry。
            // 同时排除 vprintf/assert/internal 等非业务 kernel 入口。
            if (line.find("STT_FUNC") == 0 and line.find("STO_ENTRY") != std::string::npos and
                std::none_of(illegal_names.begin(), illegal_names.end(),
                [&](const auto name) { return line.find(name) != std::string::npos; })) {
                // cuobjdump 输出的最后一个空格后通常就是符号名。
                const auto last_space = line.rfind(' ');
                // 截取并保存候选 kernel 名称。
                symbol_names.push_back(line.substr(last_space + 1));
            }
        }

        // Print symbols
        // DeepEP 的 JIT 产物预期每个 cubin 只有一个真正的 kernel entry。
        // 如果找不到或找到多个，说明缓存目录可能来自旧版本、并发写入异常或 cubin 已损坏。
        if (symbol_names.size() != 1) {
            // 打印修复建议：删除该缓存目录后重启，让 JIT 重新编译生成干净产物。
            printf("Corrupted JIT cache directory (expected 1 kernel symbol, found %zu): %s, "
                   "please run `rm -rf %s` and restart your task.\n",
                   symbol_names.size(), dir_path.c_str(), dir_path.c_str());
            // 打印当前解析出的符号列表，帮助判断是 0 个入口还是多个入口。
            printf("Symbol names: ");
            // 逐个输出候选符号名。
            for (const auto& symbol: symbol_names)
                printf("%s, ", symbol.c_str());
            // 输出换行，保持日志格式整洁。
            printf("\n");
            // 触发断言，阻止继续加载不确定的 kernel。
            EP_HOST_ASSERT(false and "Corrupted JIT cache directory");
        }

        // Load from the library
        // 加载 cubin 并根据唯一符号名取得 kernel handle；library 作为输出参数被一并填充。
        kernel = load_kernel(cubin_path, symbol_names[0], &library);
    }

    // 初始化 CUDA Toolkit 根目录。需要在构造 KernelRuntime 前由上层调用。
    static void prepare_init(const std::string& cuda_home_path_by_python) {
        // 保存 Python 侧探测到的 CUDA 路径，避免 C++ 层自行猜测环境。
        cuda_home = cuda_home_path_by_python;
    }

    // 检查 JIT 缓存目录是否可用于加载 KernelRuntime。
    static bool check_validity(const std::filesystem::path& dir_path) {
        // 目录不存在表示缓存未命中，调用方可以触发重新编译。
        if (not std::filesystem::exists(dir_path))
            return false;
        // NOTES: if the directory exists, kernel.cu and kernel.cubin must both exist,
        // because the directory is created atomically via rename
        // 目录一旦存在，按约定应是编译器原子 rename 发布的完整缓存目录。
        // 因此源码 kernel.cu 和运行产物 kernel.cubin 必须同时存在。
        if (not std::filesystem::exists(dir_path / "kernel.cu") or
            not std::filesystem::exists(dir_path / "kernel.cubin")) {
            // 缺文件说明缓存损坏，提示用户删除目录后重新运行。
            printf("Corrupted JIT cache directory (missing kernel.cu or kernel.cubin): %s, "
                   "please run `rm -rf %s` and restart your task.\n",
                   dir_path.c_str(), dir_path.c_str());
            // 对损坏缓存直接断言，而不是静默重编译，避免隐藏并发/文件系统问题。
            EP_HOST_ASSERT(false and "Corrupted JIT cache directory");
        }
        // 目录存在且关键文件齐全，可以尝试加载。
        return true;
    }

    // 析构时卸载 CUDA library/module，释放 driver 侧资源。
    ~KernelRuntime() noexcept(false) {
        // `unload_library` 可能抛出/触发错误，因此析构函数声明为 noexcept(false)。
        unload_library(library);
    }
};

// 为 KernelRuntime::cuda_home 静态成员生成定义，避免只有声明导致链接失败。
EP_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, cuda_home);

// 结束 JIT 命名空间。
} // namespace deep_ep::jit
