#!/bin/sh

set -e
cd "$(dirname "$0")"
mkdir -p models

get() {  # repo, file, saved as
  [ -f "models/$3" ] && return
  curl -fL -C - -o "models/$3.part" "https://huggingface.co/$1/resolve/main/$2"
  mv "models/$3.part" "models/$3"
}

for name in \
  gemma-2-2b-it gemma-2-9b-it gemma-2-27b-it \
  google_gemma-3-270m-it google_gemma-3-1b-it google_gemma-3-4b-it google_gemma-3-12b-it \
  gemma-4-12B-it google_gemma-4-31B-it \
  ibm-granite_granite-3.3-2b-instruct ibm-granite_granite-3.3-8b-instruct \
  ibm-granite_granite-4.1-3b ibm-granite_granite-4.1-8b \
  Mistral-Small-24B-Instruct-2501 mistralai_Mistral-Small-3.1-24B-Instruct-2503 mistralai_Mistral-Small-3.2-24B-Instruct-2506 \
  allenai_Olmo-3-7B-Instruct allenai_Olmo-3.1-32B-Instruct \
  Qwen2-0.5B-Instruct Qwen2-7B-Instruct \
  Qwen_Qwen3-0.6B Qwen_Qwen3-1.7B Qwen_Qwen3-4B \
  Qwen_Qwen3.5-0.8B Qwen_Qwen3.5-2B Qwen_Qwen3.5-4B \
  microsoft_UserLM-8b
do
  get "bartowski/$name-GGUF" "$name-Q8_0.gguf" "${name#*_}-Q8_0.gguf"
done
get Qwen/Qwen2-1.5B-Instruct-GGUF qwen2-1_5b-instruct-q8_0.gguf Qwen2-1.5B-Instruct-Q8_0.gguf
