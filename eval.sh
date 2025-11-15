 CUDA_VISIBLE_DEVICES=5,6 python eval.py --model_path /data1/liqw/checkpoints/Qwen3-8B \
    --data_dir data \
    --save_dir results \
    --max_length 4096 \
    --test_time 1 