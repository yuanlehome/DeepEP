"""
NVLink bandwidth test (standalone, only torch dependency).

Two measurements:
  1. P2P copy      -- single-process cudaMemcpyPeer, all GPU pairs (unidirectional)
  2. NCCL P2P      -- kernel-driven send+recv, true bidirectional peak per pair
  3. AllReduce     -- NCCL over all GPUs (bus bandwidth reflects NVLink aggregate)

Usage:
  python test_nvlink_bw.py
  python test_nvlink_bw.py --size-gb 2 --iters 50
  python test_nvlink_bw.py --p2p-only
  python test_nvlink_bw.py --bidir-only
  python test_nvlink_bw.py --ar-only
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ──────────────────────────────────────────────────────────────────────────────
# 1. P2P copy (single process)
# ──────────────────────────────────────────────────────────────────────────────

def _p2p_pair_bw(src: int, dst: int, n_elem: int, size_gb: float,
                 iters: int, warmup: int, bidir: bool = False) -> float:
    """Return bandwidth in GB/s for one GPU pair.

    bidir=False : unidirectional (src→dst), returns GB/s
    bidir=True  : bidirectional (src↔dst concurrently on two streams),
                  returns aggregate GB/s (= 2 × size / time per iter)
    """
    t_a = torch.ones(n_elem, dtype=torch.bfloat16, device=f"cuda:{src}")
    t_b = torch.empty(n_elem, dtype=torch.bfloat16, device=f"cuda:{dst}")

    if bidir:
        # Two independent CUDA streams, one per direction.
        # A→B runs on a stream of device dst (copy_ uses dst's current stream).
        # B→A runs on a stream of device src.
        stream_ab = torch.cuda.Stream(device=dst)   # A→B
        stream_ba = torch.cuda.Stream(device=src)   # B→A
        t_b2 = torch.ones(n_elem, dtype=torch.bfloat16, device=f"cuda:{dst}")
        t_a2 = torch.empty(n_elem, dtype=torch.bfloat16, device=f"cuda:{src}")

        def _step():
            with torch.cuda.stream(stream_ab):
                t_b.copy_(t_a)   # A→B
            with torch.cuda.stream(stream_ba):
                t_a2.copy_(t_b2) # B→A
            stream_ab.synchronize()
            stream_ba.synchronize()
    else:
        def _step():
            t_b.copy_(t_a)

    # Warmup
    for _ in range(warmup):
        _step()
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iters):
        _step()
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    elapsed_s = (time.perf_counter() - t0) / iters

    # bidir: both directions transferred size_gb each → aggregate = 2×size_gb
    return (2 * size_gb if bidir else size_gb) / elapsed_s  # GB/s


def test_p2p(size_gb: float, iters: int, warmup: int, bidir: bool = False):
    n_gpus = torch.cuda.device_count()
    n_bytes = int(size_gb * 1024 ** 3)
    n_elem = n_bytes // 2  # BF16 = 2 bytes

    mode = "Bidirectional" if bidir else "Unidirectional"
    note = " (aggregate = A→B + B→A)" if bidir else ""
    print(f"\n{'='*60}")
    print(f"P2P {mode} Bandwidth{note}  ({size_gb:.1f} GB BF16, {iters} iters)")
    print(f"{'='*60}")
    print(f"  {'pair':<12}  {'bw (GB/s)':>10}  {'P2P access'}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")

    for src in range(n_gpus):
        for dst in range(n_gpus):
            if src == dst:
                continue
            if bidir and dst < src:
                continue  # each pair tested once
            p2p_ok = torch.cuda.can_device_access_peer(src, dst)
            bw = _p2p_pair_bw(src, dst, n_elem, size_gb, iters, warmup, bidir)
            label = f"GPU{src}↔GPU{dst}" if bidir else f"GPU{src}→GPU{dst}"
            flag = "✓ (NVLink/P2P)" if p2p_ok else "✗ (PCIe fallback)"
            print(f"  {label:<12}  {bw:>10.1f}  {flag}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 2. NCCL P2P bidirectional (multi-process, kernel-driven, true bidir)
# ──────────────────────────────────────────────────────────────────────────────

def _bidir_p2p_worker(rank: int, world_size: int, size_gb: float,
                      iters: int, warmup: int):
    """Each GPU pair (src, dst) simultaneously does send+recv in both directions.

    Uses dist.batch_isend_irecv (NCCL kernel-driven), which can truly saturate
    NVLink links in both directions at the same time.
    Note: unbatched isend/irecv hits NCCL limitations; batch form is required.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29712")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    n_elem = int(size_gb * 1e9 / 2)  # BF16 = 2 bytes (SI GB for consistency)
    send_buf = torch.ones(n_elem, dtype=torch.bfloat16, device="cuda")
    recv_buf = torch.empty(n_elem, dtype=torch.bfloat16, device="cuda")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"NCCL P2P Bidirectional Bandwidth  "
              f"({size_gb:.2f} GB BF16 each dir, {iters} iters)")
        print(f"  [batch_isend_irecv: kernel-driven, truly concurrent bidir]")
        print(f"{'='*60}")
        print(f"  {'pair':<12}  {'agg (GB/s)':>11}  {'per-dir (GB/s)':>15}")
        print(f"  {'-'*12}  {'-'*11}  {'-'*15}")

    for src in range(world_size):
        for dst in range(src + 1, world_size):
            active = rank in (src, dst)
            peer = (dst if rank == src else src) if active else None

            # Warmup
            if active:
                ops = [dist.P2POp(dist.isend, send_buf, peer),
                       dist.P2POp(dist.irecv, recv_buf, peer)]
                for _ in range(warmup):
                    for req in dist.batch_isend_irecv(ops):
                        req.wait()
            dist.barrier()
            torch.cuda.synchronize()

            # Benchmark
            if active:
                t0 = time.perf_counter()
                for _ in range(iters):
                    ops = [dist.P2POp(dist.isend, send_buf, peer),
                           dist.P2POp(dist.irecv, recv_buf, peer)]
                    for req in dist.batch_isend_irecv(ops):
                        req.wait()
                torch.cuda.synchronize()
                elapsed_s = (time.perf_counter() - t0) / iters

                agg_bw = 2 * size_gb / elapsed_s
                per_dir = size_gb / elapsed_s

                if rank == src:
                    print(f"  GPU{src}↔GPU{dst}    {agg_bw:>11.1f}  {per_dir:>15.1f}")

            dist.barrier()

    dist.destroy_process_group()


def test_bidir_nccl_p2p(size_gb: float, iters: int, warmup: int):
    n_gpus = torch.cuda.device_count()
    mp.spawn(_bidir_p2p_worker, args=(n_gpus, size_gb, iters, warmup),
             nprocs=n_gpus, join=True)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 3. NCCL AllReduce (multi-process)
# ──────────────────────────────────────────────────────────────────────────────

def _ar_worker(rank: int, world_size: int, size_gb: float, iters: int, warmup: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29711")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    n_elem = int(size_gb * 1024 ** 3 / 2)  # BF16
    buf = torch.ones(n_elem, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(buf)
    dist.barrier()
    torch.cuda.synchronize()

    # Benchmark (wall clock after barrier ensures all ranks start together)
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed_s = (time.perf_counter() - t0) / iters

    if rank == 0:
        n = world_size
        algo_bw = size_gb / elapsed_s                           # GB/s
        bus_bw = 2 * size_gb * (n - 1) / n / elapsed_s        # GB/s (ring formula)
        print(f"\n{'='*60}")
        print(f"NCCL AllReduce Bandwidth  ({world_size} GPUs, {size_gb:.2f} GB BF16, {iters} iters)")
        print(f"{'='*60}")
        print(f"  time per iter : {elapsed_s*1e3:.2f} ms")
        print(f"  algo bandwidth: {algo_bw:.1f} GB/s")
        print(f"  bus  bandwidth: {bus_bw:.1f} GB/s  ← compare vs NVLink peak")
        print()

    dist.destroy_process_group()


def test_allreduce(size_gb: float, iters: int, warmup: int):
    n_gpus = torch.cuda.device_count()
    mp.spawn(_ar_worker, args=(n_gpus, size_gb, iters, warmup),
             nprocs=n_gpus, join=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Show topology
# ──────────────────────────────────────────────────────────────────────────────

def show_topology():
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "topo", "-m"],
                                      stderr=subprocess.DEVNULL).decode()
        print(f"\n{'='*60}")
        print("GPU Topology (nvidia-smi topo -m)")
        print(f"{'='*60}")
        print(out)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NVLink bandwidth test")
    parser.add_argument("--size-gb",    type=float, default=1.0,
                        help="Tensor size (GB) for P2P test (default: 1.0)")
    parser.add_argument("--ar-size-gb", type=float, default=0.25,
                        help="Tensor size (GB) for AllReduce test (default: 0.25)")
    parser.add_argument("--iters",      type=int,   default=20)
    parser.add_argument("--warmup",     type=int,   default=5)
    parser.add_argument("--p2p-only",    action="store_true")
    parser.add_argument("--bidir-only",  action="store_true",
                        help="Only run NCCL P2P bidirectional test")
    parser.add_argument("--ar-only",     action="store_true")
    parser.add_argument("--bidir",       action="store_true",
                        help="Also run copy()-based bidir after P2P test (shows CE limitation)")
    parser.add_argument("--no-topo",     action="store_true",
                        help="Skip nvidia-smi topo output")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} CUDA GPUs")
    for i in range(n_gpus):
        prop = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {prop.name}  {prop.total_memory / 1024**3:.0f} GB")

    if not args.no_topo:
        show_topology()

    if not args.ar_only and not args.bidir_only:
        test_p2p(args.size_gb, args.iters, args.warmup, bidir=False)
        if args.bidir:
            test_p2p(args.size_gb, args.iters, args.warmup, bidir=True)

    if args.bidir_only or not (args.p2p_only or args.ar_only):
        test_bidir_nccl_p2p(args.size_gb, args.iters, args.warmup)

    if not args.p2p_only and not args.bidir_only:
        test_allreduce(args.ar_size_gb, args.iters, args.warmup)

    print("Reference peaks (per GPU, unidirectional / bidirectional aggregate):")
    print("  H100  NVLink 4 : ~450 GB/s uni  ~810 GB/s bidir-agg   AllReduce bus_bw ~810 GB/s (8 GPU ring)")
    print("  B30Z  NVLink 5 : ~715 GB/s uni  ~1330 GB/s bidir-agg  AllReduce bus_bw ~1620 GB/s (8 GPU ring)")


if __name__ == "__main__":
    main()
