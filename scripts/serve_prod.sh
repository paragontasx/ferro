#!/bin/sh
# production command (2026-10-22): f32 KV + 256 MB prefix cache after the perf-fixes branch
# GPU driver stack (2026-10-24): FERRO_GPU_DRIVER=neo (default) uses Intel NEO/compute-runtime,
#   installed rootlessly at /opt/data/neo (intel-opencl-icd 26.35 + IGC 2.41); =rusticl falls back
#   to Mesa's ICD for one-variable rollback. NEO measured (same binary, single tenant):
#   350M decode 45->58 tok/s, prefill 153->275 tok/s, d1 batch parity, 1.2B decode +30-50%.
#   NEO caveat: 1.2B prefill >1k tokens is ~25% slower than rusticl (one IGC-compiled attention
#   kernel); the 350M production model wins on every metric.
cd /opt/data/ferro || exit 1
DRV="${FERRO_GPU_DRIVER:-neo}"
if [ "$DRV" = "neo" ] && [ -f /opt/data/neo/icd/intel-igdrcl.icd ]; then
  OCL_ICD_VENDORS=/opt/data/neo/icd
  LD_LIBRARY_PATH=/opt/data/neo/root/usr/local/lib:/opt/data/neo/root/usr/lib/x86_64-linux-gnu:/opt/data/neo/root/usr/lib/x86_64-linux-gnu/intel-opencl
  PATH=/opt/data/neo/root/usr/bin:$PATH
  export OCL_ICD_VENDORS LD_LIBRARY_PATH PATH
else
  unset OCL_ICD_VENDORS          # Mesa rusticl via /etc/OpenCL/vendors/rusticl.icd
fi
exec ./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --lm-slots 4 --lm-ctx 4096 --kv-dtype f32 --prefix-cache-mb 256 \
  --d1 models/d1-omni-600M-Q8_0.gguf --snapshot-dir /opt/data/ferro-snapshots --host 127.0.0.1 --port 8090
