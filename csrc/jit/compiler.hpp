#pragma once

// Linux `open` 等文件描述符相关接口，用于后面显式 fsync 文件或目录。
#include <fcntl.h>
// C++17 文件系统库：路径拼接、目录遍历、目录重命名、文件存在性检查等都依赖它。
#include <filesystem>
// 文件输出流，用于把 JIT 生成的 CUDA 源码写入临时目录。
#include <fstream>
// NVRTC 头文件。当前文件主要使用 NVCC 编译路径，但保留该依赖以配合 JIT 编译模块。
#include <nvrtc.h>
// `std::optional` 用于表示是否额外输出 PTX 文件。
#include <optional>
// 正则库，用于解析 NVCC 版本号以及检查 ptxas 输出。
#include <regex>
// 字符串类型，贯穿编译签名、命令行、源码文本和日志输出。
#include <string>

// DeepEP 的 host/device 断言宏定义，失败时会终止并报告错误。
#include <deep_ep/common/exception.cuh>

// 轻量格式化工具，类似 `fmt::format`，用于拼接编译命令和缓存路径。
#include "../utils/format.hpp"
// 哈希工具，用于把编译输入摘要成稳定的十六进制缓存 key。
#include "../utils/hash.hpp"
// 懒初始化工具，用于延迟创建全局 compiler 实例。
#include "../utils/lazy_init.hpp"
// 系统工具：读取环境变量、创建目录、调用外部命令、安全删除目录等。
#include "../utils/system.hpp"
// JIT runtime cache：根据已编译目录加载并缓存 KernelRuntime。
#include "cache.hpp"
// 设备运行时信息：查询当前 GPU 架构，决定 NVCC 的 `sm_xx` 编译目标。
#include "device_runtime.hpp"

// 所有 JIT 编译相关实现都放在 deep_ep::jit 命名空间，避免污染全局符号。
namespace deep_ep::jit {

// JIT 编译器抽象基类：负责通用缓存目录、通用编译参数、落盘同步和构建流程。
// 具体编译器只需要实现 `compile`，把给定 CUDA 源码编译成 cubin/ptx。
class Compiler {
public:
    // DeepEP Python/C++ 库根目录，由 Python 侧初始化时传入。
    static std::filesystem::path library_root_path;
    // DeepEP 的 include 目录，通常等于 `${library_root_path}/include`。
    static std::filesystem::path library_include_path;
    // CUDA Toolkit 根目录，用于定位 nvcc/cuobjdump 等工具。
    static std::filesystem::path cuda_home;
    // NCCL 根目录，用于给 JIT kernel 添加 NCCL 头文件搜索路径。
    static std::filesystem::path nccl_root;
    // cuobjdump 可执行文件路径，用于把 cubin 反汇编为 SASS。
    static std::filesystem::path cuobjdump_path;

    // 在真正创建 Compiler 实例前调用，填充所有静态路径配置。
    // 这些路径来自 Python 侧探测结果，避免 C++ 层重复猜测安装位置。
    static void prepare_init(const std::string& library_root_path,
                             const std::string& cuda_home_path_by_python,
                             const std::string& nccl_root_path_by_python) {
        // NOTES: if you are adding some third-party includes for kernels, please add its hash value
        // 记录 DeepEP 库根目录，后续派生 include 路径和缓存签名都依赖它。
        Compiler::library_root_path = library_root_path;
        // DeepEP 对外暴露的头文件目录，JIT 生成代码会通过 `-I` 引入这里。
        Compiler::library_include_path = Compiler::library_root_path / "include";
        // CUDA Toolkit 根目录，NVCCCompiler 会在 `${cuda_home}/bin` 下寻找工具链。
        Compiler::cuda_home = cuda_home_path_by_python;
        // NCCL 根目录，通用 flags 会追加 `${nccl_root}/include`。
        Compiler::nccl_root = nccl_root_path_by_python;
        // 默认 cuobjdump 路径；NVCCCompiler 构造时也会再次设置，保持与 cuda_home 一致。
        Compiler::cuobjdump_path = Compiler::cuda_home / "bin" / "cuobjdump";
    }

    // 编译器签名：用于区分不同后端/版本，例如 `NVCC12.8`。
    std::string signature;
    // 实际传给编译器的命令行参数集合，会参与缓存 key 计算。
    std::string flags;
    // JIT 缓存根目录，默认 `$HOME/.deep_ep`，可通过 `EP_JIT_CACHE_DIR` 覆盖。
    std::filesystem::path cache_dir_path;

    // 构造通用编译器状态：检查初始化、确定缓存目录、组装通用 flags。
    Compiler() {
        // 路径配置必须先由 `prepare_init` 完成，否则 JIT 编译无法定位依赖。
        EP_HOST_ASSERT(not library_root_path.empty());
        EP_HOST_ASSERT(not library_include_path.empty());
        EP_HOST_ASSERT(not cuda_home.empty());
        EP_HOST_ASSERT(not nccl_root.empty());
        EP_HOST_ASSERT(not cuobjdump_path.empty());

        // Cache settings
        // 默认把 JIT 编译产物放到用户 HOME 下，避免污染项目源码目录。
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_ep";
        // 如果用户显式指定 `EP_JIT_CACHE_DIR`，则把缓存切到该目录，便于共享或调试。
        if (const auto env_cache_dir_path = get_env<std::string>("EP_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        // 基类先给一个占位签名，派生类会覆盖成真实编译器版本。
        signature = "unknown-compiler";
        // 设置通用 C++ 标准和 NVCC 诊断抑制项；默认 C++20，可由 `EP_JIT_CPP_STANDARD` 覆盖。
        // `--register-usage-level=10` 让 ptxas 输出/记录更细粒度的寄存器使用信息。
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940,3012 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("EP_JIT_CPP_STANDARD", 20));
        // Debug 或显式要求 ptxas verbose 时，输出 ptxas 编译详情，便于调优寄存器/本地内存。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PTXAS_VERBOSE", 0))
            flags += " --ptxas-options=--verbose";
        // Debug 或显式要求 lineinfo 时，保留行号信息，方便 profiler/反汇编定位到源码。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
        // 打开 NCCL 设备侧 GIN/GDAKI 调试宏，供相关 kernel 条件编译使用。
        if (get_env("EP_GIN_GDAKI_DEBUG", 0))
            flags += " -DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1";
        // 添加 NCCL include 路径，使 JIT kernel 能包含 NCCL 设备侧头文件。
        flags += fmt::format(" -I {}/include", nccl_root.c_str());

        // Some special flags for EP
        // TODO: make it more general, e.g. `EP_JIT_EXTRA_FLAGS`
        // 可选覆盖 top-k index 位宽；非 0 时传给 CUDA 代码作为预处理宏。
        if (int num_topk_idx_bits = get_env("EP_NUM_TOPK_IDX_BITS", 0); num_topk_idx_bits != 0)
            flags += fmt::format(" -DEP_NUM_TOPK_IDX_BITS={}", num_topk_idx_bits);
    }

    // 基类存在虚函数，因此析构函数也保持 virtual，允许通过基类指针安全释放派生类。
    virtual ~Compiler() = default;

    // 返回 JIT 临时目录根路径，并确保目录存在。
    std::filesystem::path make_tmp_dir() const {
        // 所有构建先进入 `${cache_dir_path}/tmp`，构建完成后再原子迁移到 cache 目录。
        return make_dirs(cache_dir_path / "tmp");
    }

    // 对单个文件或目录执行 fsync，尽力把数据和目录项刷到后端存储。
    static void fsync_path(const std::filesystem::path& path) {
        // 使用 POSIX open 获取 fd；O_RDONLY 足够用于 fsync 已存在的文件/目录。
        const auto fd = ::open(path.c_str(), O_RDONLY);
        // 打开失败时直接跳过，调用者通常只需要 best-effort 的同步保障。
        if (fd >= 0) {
            // 刷新 fd 对应对象，避免分布式文件系统上 close 后其他节点仍不可见。
            ::fsync(fd);
            // 关闭 fd，防止文件描述符泄漏。
            ::close(fd);
        }
    }

    // Recursively fsync a directory: files and subdirectories first (bottom-up), then the directory itself
    // NOTES: ensures data and directory entries are visible on other nodes in distributed filesystems
    // 递归同步目录树：先同步子文件/子目录，再同步当前目录，保证 rename 前内容尽量可见。
    static void fsync_dir(const std::filesystem::path& dir_path) { // NOLINT(*-no-recursion)
        // 遍历当前目录下的所有直接子项。
        for (const auto& entry: std::filesystem::directory_iterator(dir_path)) {
            // 子目录需要递归到底层，确保内部文件和目录项也被同步。
            if (entry.is_directory())
                fsync_dir(entry.path());
            // 普通文件直接 fsync 内容。
            else if (entry.is_regular_file())
                fsync_path(entry.path());
        }
        // 最后 fsync 当前目录本身，确保目录项变更被持久化。
        fsync_path(dir_path);
    }

    // 把字符串数据以二进制方式写入指定路径，并在写完后 fsync。
    static void put(const std::filesystem::path& path, const std::string& data) {
        // 以 binary 模式打开，避免文本模式对换行等内容做平台相关转换。
        std::ofstream out(path, std::ios::binary);
        // 写入完整 buffer；失败会触发 DeepEP host 断言。
        EP_HOST_ASSERT(out.write(data.data(), data.size()));
        // 先关闭 C++ stream，确保缓冲区内容下沉到内核文件描述符。
        out.close();

        // NOTES: fsync to ensure the data is visible to other processes (e.g., NVCC)
        // on distributed filesystems, where `close()` alone does not guarantee persistence
        // 在分布式文件系统上，NVCC 可能运行在另一个进程/节点，显式 fsync 降低读到旧内容的概率。
        fsync_path(path);
    }

    // 构建指定名称的 JIT kernel：命中缓存则直接加载，否则编译并写入缓存。
    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        // 缓存签名包含 kernel 名称、编译器签名、完整 flags 和源码内容。
        // 任一输入变化都会生成新的 hash，避免错误复用旧 cubin。
        const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
        // 最终缓存目录形如 `${cache}/cache/kernel.<name>.<hash>`。
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        // 先查询内存级 runtime cache；如果已经加载过该目录，直接复用 KernelRuntime。
        if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Compile into a temporary directory, then atomically rename the whole directory
        // NOTES: renaming a directory is atomic on both local and distributed filesystems,
        // avoiding the stale inode issue that occurs when renaming individual files
        // 每次编译使用独立 UUID 子目录，避免多 rank/多进程同时写同一目录互相覆盖。
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        // 创建本次编译的临时工作目录。
        make_dirs(tmp_dir_path);

        // Compile into the temporary directory
        // cubin 是最终供 CUDA Driver 加载执行的二进制产物。
        const auto tmp_cubin_path = tmp_dir_path / "kernel.cubin";
        // 如果需要 dump ASM 或 PTX，则要求编译器同时生成 PTX 文件，供排查和反汇编使用。
        if (get_env<int>("EP_JIT_DUMP_ASM") or get_env<int>("EP_JIT_DUMP_PTX")) {
            // PTX 是中间表示，便于开发者检查编译前后优化效果。
            const auto tmp_ptx_path = tmp_dir_path / "kernel.ptx";
            // 调用派生类编译实现，同时输出 cubin 和 ptx。
            compile(code, tmp_dir_path, tmp_cubin_path, tmp_ptx_path);
        } else {
            // 默认只生成运行必需的 cubin，减少编译耗时和缓存占用。
            compile(code, tmp_dir_path, tmp_cubin_path);
        }

        // Disassemble if needed
        // dump ASM 或 SASS 时，用 cuobjdump 从 cubin 中提取 SASS 汇编。
        if (get_env<int>("EP_JIT_DUMP_ASM") or get_env<int>("EP_JIT_DUMP_SASS")) {
            // SASS 是目标 GPU 的机器级汇编，常用于分析指令、寄存器和访存。
            const auto tmp_sass_path = tmp_dir_path / "kernel.sass";
            // 反汇编 cubin 并写入临时目录。
            disassemble(tmp_cubin_path, tmp_sass_path);
        }

        // Fsync before rename to ensure visibility on distributed filesystems
        // 在发布到最终缓存目录前同步整个临时目录，避免其他 rank 看到目录但读不到完整文件。
        fsync_dir(tmp_dir_path);

        // Atomically rename the temporary directory to the final cache path
        // NOTES: if another rank already created dir_path, rename will fail — that's fine
        // 先确保最终 cache 父目录存在。
        make_dirs(dir_path.parent_path());
        // 使用 error_code 版本避免 rename 失败直接抛异常，方便处理多进程竞争。
        std::error_code error_code;
        // 原子目录重命名：成功则当前进程发布缓存；失败通常说明其他进程已发布同 hash 缓存。
        std::filesystem::rename(tmp_dir_path, dir_path, error_code);
        // 如果最终目录已存在或 rename 遇到文件系统竞争，则丢弃自己的临时目录。
        if (error_code) {
            // Another rank beat us, then clean up our dir and use the existing one
            // NOTES: avoid `std::filesystem::remove_all` here — it can segfault on
            // distributed filesystems, when concurrent processes operate
            // on the same parent directory, causing stale directory entries
            // 使用项目内安全删除实现，规避分布式文件系统并发 remove_all 的问题。
            safe_remove_all(tmp_dir_path);
        }

        // Put into the runtime cache
        // 从最终缓存目录加载 KernelRuntime；若本进程 rename 失败，则这里加载其他 rank 生成的产物。
        const auto runtime = kernel_runtime_cache->get(dir_path);
        // 加载失败说明编译产物不完整或目录不可见，需要立即暴露错误。
        EP_HOST_ASSERT(runtime != nullptr);
        // 返回可执行 runtime，调用方随后可启动 kernel。
        return runtime;
    }

    // 使用 cuobjdump 把 cubin 文件反汇编成 SASS 文本。
    static void disassemble(const std::filesystem::path &cubin_path, const std::filesystem::path &sass_path) {
        // Disassemble the CUBIN file to SASS
        // 通过 shell 重定向把 `--dump-sass` 的输出写入目标 sass 文件。
        const auto command = fmt::format("{} --dump-sass {} > {}", cuobjdump_path.c_str(), cubin_path.c_str(), sass_path.c_str());
        // Debug 或显式打印命令时，把实际执行的 cuobjdump 命令输出到 stdout。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running cuobjdump command: %s\n", command.c_str());
        // 调用外部命令并捕获返回码和输出。
        const auto [return_code, output] = call_external_command(command);
        // cuobjdump 非 0 返回码表示反汇编失败，打印输出后触发断言。
        if (return_code != 0) {
            printf("cuobjdump failed: %s\n", output.c_str());
            EP_HOST_ASSERT(false and "cuobjdump failed");
        }
    }

    // 派生类必须实现的编译入口：把 `code` 写入/编译到 `dir_path`，生成 cubin，可选生成 ptx。
    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::optional<std::filesystem::path> &ptx_path = std::nullopt) const = 0;
};

// 为 Compiler 类中的静态路径成员生成定义，避免只有声明导致链接失败。
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, nccl_root);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuobjdump_path);

// 基于 NVIDIA NVCC 的具体 JIT 编译器实现。
class NVCCCompiler final: public Compiler {
    // nvcc 可执行文件路径，默认 `${cuda_home}/bin/nvcc`，可通过环境变量覆盖。
    std::filesystem::path nvcc_path;

    // 调用 `nvcc --version` 并解析主/次版本号。
    std::pair<int, int> get_nvcc_version() const {
        // 在运行外部命令前确认 nvcc 路径真实存在。
        EP_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        // 拼出版本查询命令。
        const auto command = std::string(nvcc_path) + " --version";
        // 执行命令并捕获输出，输出中包含 `release X.Y` 字段。
        const auto [return_code, output] = call_external_command(command);
        // nvcc --version 必须成功，否则无法可靠生成编译器签名和架构参数。
        EP_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3
        // 存放解析得到的主版本和次版本。
        int major, minor;
        // 正则匹配结果容器。
        std::smatch match;
        // 从 nvcc 输出中提取 `release 12.8` 这类版本字符串。
        EP_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        // 把 `X.Y` 拆成整数 major/minor。
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        // DeepEP JIT 要求 NVCC 至少 12.3，低版本可能缺少所需 CUDA/C++ 能力。
        EP_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        // 返回版本元组，供构造函数设置签名和判断架构后缀支持。
        return {major, minor};
    }

public:
    // 构造 NVCC 编译器：定位工具链、解析版本、追加 NVCC 专属 flags。
    NVCCCompiler() {
        // Override the compiler signature
        // 默认从 CUDA Toolkit 目录定位 nvcc。
        nvcc_path = cuda_home / "bin" / "nvcc";
        // 同步设置 cuobjdump 路径，保证反汇编工具与 nvcc 来自同一个 CUDA Toolkit。
        cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        // 允许用户通过 `EP_JIT_NVCC_COMPILER` 指向自定义 nvcc，例如多 CUDA 版本环境。
        if (const auto env_nvcc_path = get_env<std::string>("EP_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        // 获取 nvcc 版本，用于缓存签名和能力判断。
        const auto [nvcc_major, nvcc_minor] = get_nvcc_version();
        // 编译器版本进入签名，升级/切换 nvcc 后会自动生成不同缓存目录。
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // The override the compiler flags
        // Only NVCC >= 12.9 supports arch-specific family suffix
        // 查询当前设备架构；NVCC 12.9+ 才允许使用带 family suffix 的架构表达。
        const auto arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        // 在基类 flags 基础上追加 DeepEP include、目标 GPU 架构和 NVCC/C++ 优化参数。
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), arch);
    }

    // NVCC 后端编译实现：生成 `.cu`，调用 nvcc 输出 cubin，并按需输出 PTX。
    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        // JIT 生成源码统一命名为 kernel.cu，放入本次构建临时目录。
        const auto code_path = dir_path / "kernel.cu";
        // 写入源码并 fsync，确保随后启动的 nvcc 进程能读到完整文件。
        put(code_path, code);

        // Compile to CUBIN
        // Avoid cwd files shadowing C++ standard library headers
        // 在干净的临时目录中运行 nvcc，避免当前工作目录里同名头文件遮蔽系统/C++ 标准库头。
        const auto compile_dir = make_tmp_dir();
        // 组装 cubin 编译命令：`nvcc kernel.cu -cubin -o kernel.cubin <flags>`。
        const auto command = fmt::format("cd {} && {} {} -cubin -o {} {}",
            compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        // 按需打印完整 NVCC 命令，便于复现 JIT 编译问题。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        // 执行 NVCC 并捕获 ptxas/nvcc 输出。
        const auto [return_code, output] = call_external_command(command);
        // NVCC 失败时打印完整输出，再通过断言中止。
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            EP_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Compile to PTX if needed
        // 如果调用方传入 ptx_path，说明需要额外保留 PTX 供 dump/debug。
        if (ptx_path.has_value()) {
            // 组装 PTX 编译命令：同一份源码和 flags，但目标从 cubin 改为 ptx。
            const auto ptx_command = fmt::format("cd {} && {} {} -ptx -o {} {}",
                compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), ptx_path->c_str(), flags);
            // 按需打印 PTX 编译命令。
            if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
                printf("Running NVCC PTX command: %s\n", ptx_command.c_str());
            // 执行 PTX 编译并捕获输出。
            const auto [ptx_return_code, ptx_output] = call_external_command(ptx_command);
            // PTX 编译失败同样视为 JIT 构建失败。
            if (ptx_return_code != 0) {
                printf("NVCC PTX compilation failed: %s\n", ptx_output.c_str());
                EP_HOST_ASSERT(false and "NVCC PTX compilation failed");
            }
        }

        // Check local memory usage
        // 可选检查 ptxas 输出中是否出现本地内存使用；出现通常意味着寄存器溢出或栈/数组落本地内存。
        if (get_env("EP_JIT_PTXAS_CHECK", 0))
            EP_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        // Debug 或 verbose 模式下打印 NVCC/ptxas 输出，辅助分析寄存器、spill 和 occupancy。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

// 全局懒初始化编译器实例：第一次使用时创建 NVCCCompiler，之后复用同一个对象。
static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    // 当前实现固定使用 NVCC 后端；未来如增加 NVRTC/clang 后端，可在这里按环境变量选择。
    return std::make_shared<NVCCCompiler>();
});

// 结束 JIT 命名空间。
} // namespace deep_ep::jit
