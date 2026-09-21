# Recorded reference

`k3-pruned.parquet` holds what `manifests/k3-tp4-prefill-16k.json` predicts
over `corpus.json`: at the last `tail + 1` positions of every prompt, the
top-20 token ids with their log probabilities and the log probability of
the token the corpus has there. A candidate manifest is judged against it
with nothing else loaded (`docs/test.md` in kern, "录好的参考").

```sh
# the corpus: 48 prompts of 6000 characters (prose, code, licenses), tail 16
python3 reference/make_corpus.py <text dirs...>

# record the reference (once, on a free 4-GPU tray, in the environment the judge will run in)
kern test manifests/k3-tp4-prefill-16k.json --record reference/k3-pruned.parquet \
  --corpus reference/corpus.json --kernels build --weights <checkpoint> --tokenizer <tokenizer.json> --gpu 0,1,2,3

# judge a candidate (the loop's command); --prompts N for a quick pass
kern test manifests/<candidate>.json --reference reference/k3-pruned.parquet \
  --kernels build --weights <checkpoint> --gpu 0,1,2,3 --out results/<candidate>.json
```

One row per token position (`prompt`, `pos`, `id`); a producer that
scored a position adds a row with `producer`, `ref_logprob`, `top_ids`,
`top_logprob`. `tail` is file metadata. Another implementation's scores
go into the same file as another producer; with two or more producers the
verdict is read against the band of their mutual disagreement instead of
the fixed KL limit.

## Record where you judge

The file is recorded inside the agent's evaluation image (CUDA 13.0,
cuBLASLt 13.0.2.14). An earlier recording made on a host with cuBLASLt
13.7.0.10 failed the default manifest itself when judged in the image: KL
p50 6e-3, p99 0.09, one argmax flip at margin 0.18 on 816 real-text
positions. A cuBLASLt version bump changes which GEMM algorithms run and
that alone moves the distribution by more than the 1e-2 gate. So the
reference is recorded in the environment the judge runs in, and the first
thing to check when a candidate fails is whether the default passes.

## What the recording showed (2026-09-21, four GB300, in the image)

| step | time | result |
|---|---|---|
| record the default, 48 prompts × (1 prefill + 16 decode) = 816 positions | 40.9 s after a 15 s load | `k3-pruned.parquet`, 424 KB, producer `k3-tp4-prefill-16k` |
| the default against its own file | 41.3 s | PASS, KL ≤ 2.5e-12, 816/816 argmax agree |
| `lat_down` landing bf16 directly (the adopted default) vs the f32 original, `kern test` A/B | | bit-identical at all 736 spans, `next_token` identical on four ranks |

Sanity of the file (pandas): median `ref_logprob` −0.57, argmax equals the
corpus token at 60.7% of positions, median top-1/top-2 margin 2.1 nats.
