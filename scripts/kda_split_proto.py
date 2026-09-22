"""Sequence-split KDA prefill on top of the unmodified flash_kda kernel (vLLM FlashKDA dev b59532f1 as
a torch extension, e.g. `pip install` of the fork; the kern-patched K2 can be put first on PYTHONPATH).

The recurrence is affine in its state, S_end = S_start * M + C, and the output is affine in the
start state too, so one 16k sequence becomes P varlen pieces run twice, both passes parallel
over pieces: pass A with the real values from a zero start (local outputs, C_p), pass B with
v = 0 from an identity start (per-token z_t = M_{<t} q_t as its output, M_p as its final state).
Chaining E_p = E_{p-1} M_p + C_p is P small matmuls; the fix-up is out += E_{p-1} z per piece.

kda_split_proto.py <H> <T> <P> [iters]   (REF=1 also compares against SGLang's Triton chunk_kda)
"""
import sys
import time
import torch
import flash_kda

H, T, P = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20
dev, K = 'cuda', 128
gen = torch.Generator(device=dev).manual_seed(0)
r = lambda *s: torch.randn(*s, generator=gen, device=dev, dtype=torch.float32)
q, k = (r(1, T, H, K).to(torch.bfloat16) for _ in range(2))
g = (-8 + 0.5 * r(1, T, H, K)).to(torch.bfloat16)   # log-decay ~ -5*sigmoid(-8) = -1.7e-3/token: state lives ~1k tokens, as in K3
v = (0.5 * r(1, T, H, K)).to(torch.bfloat16)
beta = r(1, T, H).to(torch.bfloat16)
A_log, dt_bias = 0.3 * r(H), 0.1 * r(H, K)
S0 = 0.05 * r(1, H, K, K)
scale, lb = K ** -0.5, -5.0
zeros_v = torch.zeros_like(v)
cu1 = torch.tensor([0, T], device=dev, dtype=torch.int64)
bounds = [round(T * p / P / 16) * 16 for p in range(P + 1)]   # pieces are whole 16-token chunks
cuP = torch.tensor(bounds, device=dev, dtype=torch.int64)
ws = torch.empty(flash_kda.get_workspace_size(T, H, P), dtype=torch.uint8, device=dev)


def single():
    out, fin = torch.empty_like(v), torch.empty_like(S0)
    flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lb, S0, fin, cu1, ws)
    return out, fin


eye = torch.eye(K, device=dev).expand(P, H, K, K).contiguous()
initA = torch.zeros(P, H, K, K, device=dev)
initA[0] = S0[0]


def split():
    outA, C = torch.empty_like(v), torch.empty(P, H, K, K, device=dev)
    z, M = torch.empty_like(v), torch.empty(P, H, K, K, device=dev)
    flash_kda.fwd(q, k, v, g, beta, scale, outA, A_log, dt_bias, lb, initA, C, cuP, ws)
    flash_kda.fwd(q, k, zeros_v, g, beta, scale, z, A_log, dt_bias, lb, eye, M, cuP, ws)
    E = C[0]                                   # true end state of piece 0 (it started from S0)
    Eprev = torch.empty(P - 1, H, K, K, device=dev)
    for p in range(1, P):
        Eprev[p - 1] = E
        E = E @ M[p] + C[p]
    # fix-up: out_t += E_{p-1} z_t for every token of pieces 1..P-1, one bf16 batched GEMM per piece
    for p in range(1, P):
        lo, hi = bounds[p], bounds[p + 1]
        zt = z[0, lo:hi].transpose(0, 1)                                   # [H, Tp, K]
        outA[0, lo:hi] += torch.bmm(zt, Eprev[p - 1].to(torch.bfloat16).transpose(1, 2)).transpose(0, 1)
    return outA, E.unsqueeze(0)


out_ref, fin_ref = single()
out_s, fin_s = split()
err = (out_s.float() - out_ref.float()).abs()
print(f"H={H} T={T} P={P}: out max|err| {err.max():.3e} (ref max {out_ref.float().abs().max():.2f}, "
      f"rel {err.max() / out_ref.float().abs().max():.2e}), final state max|err| "
      f"{(fin_s - fin_ref).abs().max():.3e} (ref max {fin_ref.abs().max():.2f})")
for name, fn in [('single', single), ('split', split)]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    print(f"  {name}: {(time.perf_counter() - t0) / iters * 1e6:.0f} us")


def gpu_ms(fn, tag):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    ks = [(e.key[:34], e.device_time_total) for e in prof.key_averages() if e.device_time_total > 0]
    ks.sort(key=lambda x: -x[1])
    print(f"  {tag} GPU kernel time: {sum(t for _, t in ks):.0f} us: " + ", ".join(f"{n} {t:.0f}" for n, t in ks[:5]))


gpu_ms(single, 'single')
gpu_ms(split, 'split')
outA, C = torch.empty_like(v), torch.empty(P, H, K, K, device=dev)
gpu_ms(lambda: flash_kda.fwd(q, k, v, g, beta, scale, outA, A_log, dt_bias, lb, initA, C, cuP, ws), 'pass A only')

if __import__('os').environ.get('REF'):
    # Triton chunk_kda (fp32 state, in-place state slots) as the reference for both paths.
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    for label, st in [('[H,V,K]', S0), ('[H,K,V]', S0.transpose(-1, -2).contiguous())]:
        slots = st.clone()
        try:
            res = chunk_kda(q=q, k=k, v=v, g=g, beta=beta.float().sigmoid().to(torch.bfloat16), initial_state=slots,
                            initial_state_indices=torch.zeros(1, device=dev, dtype=torch.int32),
                            use_qk_l2norm_in_kernel=True, cu_seqlens=cu1, A_log=A_log, dt_bias=dt_bias.reshape(-1),
                            lower_bound=lb)
        except Exception as exc:
            print('  ref', label, 'failed:', str(exc)[-600:]); continue
        o_ref = res[0] if isinstance(res, (tuple, list)) else res
        e1 = (out_ref.float() - o_ref.float()).abs().max().item()
        e2 = (out_s.float() - o_ref.float()).abs().max().item()
        f_ref = slots if label == '[H,V,K]' else slots.transpose(-1, -2)
        print(f"  ref state layout {label}: out single vs triton max|err| {e1:.3e}, split vs triton {e2:.3e} "
              f"(ref out max {o_ref.float().abs().max():.2f}); final state single {(fin_ref - f_ref).abs().max():.3e}, "
              f"split {(fin_s - f_ref).abs().max():.3e}")
