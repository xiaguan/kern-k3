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
