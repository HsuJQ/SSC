export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_USE_V1=0

# Source Ascend environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

HOST=0.0.0.0
PORT=1043  


LOCAL_CKPT_DIR=/opt/pangu/Qwen--Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

SERVED_MODEL_NAME=qwen2.5_7b_instruct


vllm serve $LOCAL_CKPT_DIR \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --max-num-seqs 32 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --dtype bfloat16 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
