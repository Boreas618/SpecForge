#!/usr/bin/env python3
"""Compare flex_attention backends (default Triton vs FA4) for the DSpark-V4 draft
attention shape on Blackwell: q=[B,64,Q,512], kv=[B,1,S+Q,512] (K==V, GQA 64:1),
dual-source + sliding-window(128) block mask. Measures fwd/bwd + correctness."""
import time, torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention import sdpa_kernel, SDPBackend, activate_flash_attention_impl

dev = "cuda:0"
B, H, Q, S, D = 2, 64, 2560, 4096, 512
bs = 5; N = Q // bs; sw = 128
torch.manual_seed(0)
q = torch.randn(B, H, Q, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
kv = torch.randn(B, 1, S + Q, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
scale = 1.0 / (D ** 0.5)
anchors = torch.randint(sw + 1, S - bs, (B, N), device=dev).sort(dim=1).values
keep = torch.ones(B, N, dtype=torch.bool, device=dev)

def mask_mod(b, h, q_idx, kv_idx):
    qb = q_idx // bs
    safe_qb = qb.clamp(max=N - 1)
    anchor = anchors[b, safe_qb]
    kp = keep[b, safe_qb]
    p_abs = anchor + (q_idx - qb * bs)
    ctx = (kv_idx < S) & (kv_idx < anchor) & (kv_idx > (p_abs - sw))
    draft = (kv_idx >= S) & (qb == (kv_idx - S) // bs)
    return (ctx | draft) & kp & (qb < N)

def bench(fn, name, iters=8):
    for _ in range(3):  # warmup / compile
        q.grad = None; kv.grad = None
        o = fn(); o.sum().backward()
    torch.cuda.synchronize()
    # fwd
    t = time.time()
    for _ in range(iters):
        with torch.no_grad(): o = fn()
    torch.cuda.synchronize(); fwd = (time.time() - t) / iters * 1000
    # fwd+bwd
    t = time.time()
    for _ in range(iters):
        q.grad = None; kv.grad = None
        o = fn(); o.sum().backward()
    torch.cuda.synchronize(); fb = (time.time() - t) / iters * 1000
    print(f"{name:24s} fwd={fwd:7.2f}ms  fwd+bwd={fb:7.2f}ms")
    return o

# --- default Triton flex, BLOCK_SIZE 32 (a1's config) ---
bm32 = create_block_mask(mask_mod, B=B, H=None, Q_LEN=Q, KV_LEN=S + Q, device=dev, BLOCK_SIZE=(32, 32))
flex_c = torch.compile(flex_attention)
ko32 = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32}
try:
    o_triton = bench(lambda: flex_c(q, kv, kv, block_mask=bm32, scale=scale, enable_gqa=True, kernel_options=ko32), "flex-triton(32,32)")
except Exception as e:
    print("triton flex FAILED:", repr(e)[:160]); o_triton = None

# --- FA4 backend for flex ---
print("\nactivating FA4 impl for flex...")
try:
    activate_flash_attention_impl("FA4")
    bm = create_block_mask(mask_mod, B=B, H=None, Q_LEN=Q, KV_LEN=S + Q, device=dev)  # default block size
    flex_c2 = torch.compile(flex_attention)
    def fa4_fn():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return flex_c2(q, kv, kv, block_mask=bm, scale=scale, enable_gqa=True)
    o_fa4 = bench(fa4_fn, "flex-FA4")
    if o_triton is not None:
        d = (o_triton.float() - o_fa4.float()).abs()
        print(f"FA4 vs triton: max_abs_diff={d.max().item():.3e} mean={d.mean().item():.3e}")
except Exception as e:
    import traceback; traceback.print_exc()
    print("FA4 flex FAILED:", repr(e)[:200])
