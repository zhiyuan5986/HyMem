model_name="minimax-m2.5" 
model_name="deepseek-chat"
# base_url="http://localhost:8000/v1"
base_url="https://models.sjtu.edu.cn/api/v1"
# api_key="sk-Of0btqIp3GW0Jv65PhkqbQ" # openclaw
api_key="sk-tYfudtgpHIqCKpxWisXgtA" # sjtu

export OPENAI_API_KEY=$api_key
export OPENAI_BASE_URL=$base_url

python scripts/consolidate_locomo.py \
  --logs-dir scripts/cached_memories_advanced_deepseek-chat_0.5_20260509105918/memory_exports \
  --db-search-roots scripts/cached_memories_advanced_deepseek-chat_0.5_20260509105918 \
  --compressor-model-name ~/Documents/gpt2-dolly \
  --compressor-device-map cuda \
  --top-k 3 \
  --turn-window-k 3 \
  --condition-in-question after \
  --condition-text "Please focus on facts related to this memory entry." \
  --condition-placement prepend \
  --model $model_name \
  --openai-base-url $base_url \
  --openai-api-key $api_key \
  --embed-model ~/Documents/Qwen3-Embedding-0.6B