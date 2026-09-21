# Recorded reference

`k3-pruned.parquet` holds what `manifests/k3-tp4-prefill-16k.json` predicts
over `corpus.json`: at the last `decode + 1` positions of every prompt, the
top-20 token ids with their log probabilities and the log probability of
the token the corpus has there. A candidate manifest is judged against it
with nothing else loaded (`docs/test.md` in kern, "录好的参考").

```sh
# the corpus: 48 prompts of 6000 characters (prose, code, licenses), decode 16
python3 reference/make_corpus.py <text dirs...>

# record the reference (once, on a free 4-GPU tray)
kern test manifests/k3-tp4-prefill-16k.json --record reference/k3-pruned.parquet \
  --corpus reference/corpus.json --kernels build --weights <checkpoint> --tokenizer <tokenizer.json> --gpu 0,1,2,3

# judge a candidate (the loop's command); --prompts N for a quick pass
kern test manifests/<candidate>.json --reference reference/k3-pruned.parquet \
  --kernels build --weights <checkpoint> --gpu 0,1,2,3 --out results/<candidate>.json
```

One row per token position (`prompt`, `pos`, `token`); a producer that
scored a position adds a row with `producer`, `ref_logprob`, `top_ids`,
`top_logprob`. `decode` is file metadata. Another implementation's scores
go into the same file as another producer; with two or more producers the
verdict is read against the band of their mutual disagreement instead of
the fixed KL limit.

## What the first recording showed (2026-09-21, four GB300)

| step | time | result |
|---|---|---|
| record the default manifest, 48 prompts × (1 prefill + 16 decode) = 816 positions | 40.6 s after a 70 s load | `k3-pruned.parquet`, 424 KB |
| the default against its own file | 41 s | PASS, KL ≤ 2.5e-12, 816/816 argmax agree |
| `k3-tp4-prefill-16k-l0-bf16.json` regenerated from the current default (only `l0.wgu` on `gemm_bf16`) | 41 s | PASS, identical to the reference: `kern_k3_land_situ` already lands the f32 partial to bf16 before the activation, so a bf16-output GEMM moves the same rounding, nothing else; `kern bench` 16k over 0 687.2 ms vs 687.4 ms |
| the candidate as a second producer, judged against both | 41 s | PASS within the band (KL floor 1e-6) |
| the previously committed l0-bf16 file (built on the pre-upgrade default: old FlashKDA, v1 residual / out-gate / span-gather) | 0.9 s | FAIL at the first confident flip (prompt 0 pos 1463, margin 0.19, KL 3.9e-2); with `--logit-kl 1`: 59/816 argmax flips, all at margin ≤ 0.58, KL p50 1.3e-2 prefill / 9.3e-3 decode, p99 0.12 / 0.23, max 0.66 |

The last row is the reason this file exists. Those four kernel upgrades passed
the A/B gate as bit-identical spans or KL ≤ 1.8e-3 on the eight seeded rows of
a 16k random-token prefill; on 816 real-text positions they move the
distribution by an order of magnitude more. Whether that is acceptable cannot
be read from one producer. The anchor to add next is the model authors' own
implementation (`modeling_kimi_k3.py` in the checkpoint, run once in torch),
recorded into this file as a second producer, and a small task set for the
loop's winners.

Sanity of the file (pandas): median `ref_logprob` −0.57, argmax equals the
corpus token at 60.7% of positions, median top-1/top-2 margin 2.1 nats.
