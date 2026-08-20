export ASCEND_VISIBLE_DEVICES=5
vllm serve /nfs-share/hf_weights/Qwen3-8B \
    --dtype bfloat16 \
    --max_model_len 16384 \
    --max_num_seqs 1 \
    --gpu_memory_utilization 0.6 \
    --port 8007 \
    -tp 1 \
    --compilation_config '{"cudagraph_mode": "NONE"}' \
    --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"./vllm_profile\", \"torch_profiler_record_shapes\": true, \"torch_profiler_with_memory\": true}"\
    --speculative-config '{"model": "/nfs-share/hf_weights/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"}