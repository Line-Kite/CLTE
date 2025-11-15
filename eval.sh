 CUDA_VISIBLE_DEVICES=0 python eval.py --model_path Qwen/Qwen3-8B \
    --data_dir data \
    --save_dir results \
    --max_length 4096 \
    --test_time 3 