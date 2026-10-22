#!/bin/sh
# production command (2026-10-22): f32 KV + 256 MB prefix cache after the perf-fixes branch
cd /opt/data/ferro || exit 1
exec ./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --lm-slots 4 --lm-ctx 4096 --kv-dtype f32 --prefix-cache-mb 256 --no-prefix-tail \
  --d1 models/d1-omni-600M-Q8_0.gguf --snapshot-dir /opt/data/ferro-snapshots --host 127.0.0.1 --port 8090
