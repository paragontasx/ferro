#!/bin/sh
# Download the d1-omni model + projector and the small media files used by the parity tests.
set -e
cd "$(dirname "$0")/.."
mkdir -p models bench/d1/media
HF=https://huggingface.co/LiquidAI/d1-omni-600M-GGUF/resolve/main
[ -f models/d1-omni-600M-Q8_0.gguf ] || curl -L -o models/d1-omni-600M-Q8_0.gguf $HF/d1-omni-600M-Q8_0.gguf
[ -f models/mmproj-d1-omni-600M-Q8_0.gguf ] || curl -L -o models/mmproj-d1-omni-600M-Q8_0.gguf $HF/mmproj-d1-omni-600M-Q8_0.gguf
echo "mmproj sha256 expected: df887978550cdbb45f0f57d7b436a0f844689f9f561395a144ced500606c3a89"
sha256sum models/mmproj-d1-omni-600M-Q8_0.gguf
[ -f bench/d1/media/cats.jpg ] || curl -L -o bench/d1/media/cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg
[ -f bench/d1/media/jfk.wav ] || curl -L -o bench/d1/media/jfk.wav https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav
echo "EmbeddingGemma GGUF: place it in models/ (see README). Derived media (tiny/big png, mp3, flac, long.wav) are made with ffmpeg; see bench/mm/README or RESULTS.md."
