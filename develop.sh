# Change current directory into project root
original_dir=$(pwd)
export NVSHMEM_DIR="${original_dir}/libnvshmem-linux-x86_64-3.4.5_cuda13-archive"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=10.3a
export EP_JIT_CACHE_DIR="${original_dir}/.deep_ep_cache"
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py build

# Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" deep_ep/
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi
# 在 develop.sh 的 python setup.py build 之后添加
envs_file=$(find build -name "envs.py" -path "*/deep_ep/envs.py" | head -n 1)
if [ -n "$envs_file" ]; then
    cp "$envs_file" deep_ep/envs.py
fi

# Open users' original directory
cd "$original_dir"
