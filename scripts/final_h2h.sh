#!/bin/bash
# Final interleaved head-to-head: ferro (cpu / igpu / hetero) vs llama.cpp (cpu / vulkan igpu), same tokens.
cd /opt/data/ferro
M=models/embeddinggemma-2-Q8_0.gguf
run() { sleep 8; echo -n "$1 | "; shift; "$@" 2>&1 | grep -E "BEST|rep [0-9]+:" | tail -1; }
for W in 16 64; do
  TOK=bench/tokens_16.txt; SEN=bench/sentences.txt
  [ $W = 64 ] && TOK=bench/tokens_64.txt && SEN=bench/sentences_64.txt
  for round in 1 2; do
    echo "=== workload ${W} seqs, round $round ==="
    run "ferro cpu (2t)        " ./ferro-embed bench --model $M --tokens $TOK --reps 8 --cpu
    run "ferro igpu            " ./ferro-embed bench --model $M --tokens $TOK --reps 8
    run "ferro hetero          " ./ferro-embed bench --model $M --tokens $TOK --reps 8 --hetero
    run "llama.cpp cpu (2t)    " env REF_NGL=0 REF_THREADS=2 ./bench/ferro_ref_vk $M $SEN /tmp/x.json --time 8
    run "llama.cpp vulkan igpu " env GGML_VK_VISIBLE_DEVICES=0 REF_NGL=99 ./bench/ferro_ref_vk $M $SEN /tmp/x.json --time 8
  done
done
