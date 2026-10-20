# bifrost — IPC benchmark

Origin: https://github.com/lwa-project/bifrost

[Bifrost](https://github.com/lwa-project/bifrost) is a stream-processing
framework built around a ring-buffer pipeline. Here it's used purely as one
of the IPC mechanisms under benchmark: `benchmark.py` builds a two-block
pipeline (a `SourceBlock` producer and a `TransformBlock` consumer) and
times message delivery through bifrost's ring buffer.

This folder only contains the benchmark harness (`benchmark.py`,
`requirements.txt`) — the actual bifrost source is fetched and built
automatically (CPU-only, no CUDA needed) by the top-level `bootstrap.sh`
into `vendor/bifrost-src/`, which is gitignored, not committed.

## Install

From the repo root:

```bash
./bootstrap.sh --only bifrost
```

This creates `bifrost/venv` and, before installing bifrost from
`requirements.txt`, clones the pinned bifrost commit and builds it
(`./configure --disable-cuda && make`) so its Python bindings can be
installed.

## Run the benchmark

```bash
cd bifrost
venv/bin/python3 benchmark.py \
  --n_messages 1000 \
  --num_channels 2 \
  --msg_size 3 \
  --t_wait 0.0001 \
  --overwrite
```

- `--n_messages`: number of timed messages sent (there's also a fixed
  500-message warm-up phase before timing starts)
- `--num_channels` / `--msg_size`: shape of each message (doubles)
- `--t_wait`: seconds the producer sleeps between sends (0 = no wait)
- `--overwrite`: rerun even if results for this channel/size combination
  already exist

Results are written to `rt_results/<num_channels>_<msg_size>_{Producer,Consumer}.csv`.
