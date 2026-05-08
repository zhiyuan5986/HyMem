OPENAI_API_KEY=sk-tYfudtgpHIqCKpxWisXgtA 
OPENAI_BASE_URL=https://models.sjtu.edu.cn/api/v1 
python scripts/evaluate_locomo.py \
 --dataset ./data/locomo10.json \
 --backend openai \
 --model_name deepseek-chat \
 --embed_model ~/Documents/Qwen3-Embedding-0.6B \
 --api_key sk-tYfudtgpHIqCKpxWisXgtA \
 --embed_api_key sk-tYfudtgpHIqCKpxWisXgtA \
 --base_url https://models.sjtu.edu.cn/api/v1 \
 --embed_base_url https://models.sjtu.edu.cn/api/v1 \
 --retrieve_k 10 \
 --retrieve_k_rough 30 \
 --temperature 0.5 \
 --log_name locomo_lancedb
