# 盘古模型推理指南

## 概述

当前软件配置如下：
- CANN: 8.1.rc1
- vllm: 0.9.2
- vllm-ascend: 0.9.2rc1
- torch: 2.5.1
- torch_npu: 2.5.1.post1

并且在/opt/panguy目录下已经预下载盘古模型权重：
- 盘古1B：/opt/pangu/openPangu-Embedded-1B-V1.1
- 盘古7B：/opt/pangu/openPangu-Embedded-7B-V1.1

## 7B模型推理

- 设置环量

如果您是基于conda环境进行推理（非docker镜像），请先切换到`pangu`环境:

```bash
conda activate pangu
```

然后设置相应环境变量：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

- 查看服务器ip，后面步骤都假设查询的ip为`192.168.0.233`
- 编写拉起推理脚本: run_service_7b.sh

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 # 根据实际情况进行修改, 此样例采用4卡推理
export VLLM_USE_V1=1
HOST=192.168.0.233 # 根据实际情况进行修改
PORT=1040          # 根据实际情况进行修改
LOCAL_CKPT_DIR=/opt/pangu/openPangu-Embedded-7B-V1.1
SERVED_MODEL_NAME=pangu_embedded_7b
vllm serve $LOCAL_CKPT_DIR \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --max-num-seqs 32 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --tokenizer-mode "slow" \
    --dtype bfloat16 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
```

- 拉起服务脚本
```bash
nohup bash run_service_7b.sh > run_7b.log 2>&1 &
```

出现如下打印信息表示拉起成功
```txt
INFO:     Started server process [3135079]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

- 编写测试脚本: test_7b.sh

```bash
MASTER_NODE_IP=192.168.0.233  # server node ip
PORT=1040
SERVED_MODEL_NAME=pangu_embedded_7b
curl http://${MASTER_NODE_IP}:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$SERVED_MODEL_NAME'",
        "messages": [
            {
                "role": "user",
                "content": "How to keep healthy. Response in Chinese!"
            }
        ],
        "max_tokens": 512,
        "temperature": 0
    }'
```

- 运行测试脚本

```bash
bash test_7b.sh
```

运行结果样例：
```
{"id":"chatcmpl-df03f8d8fbfe49b3a8c26024700932e5","object":"chat.completion","created":1758703433,"model":"pangu_embedded_7b","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":" [unused16] 好的，用户问的是“How to keep health
y. Response in Chinese!”，需要我用中文回答。首先，我需要确定用户的具体需求。他们可能想要一些保持健康的实用建议，可能包括饮食、运动、作息等方面。用户可能希望得到全面但简洁的信息，适合快速阅读。\n\n接下来，我要考虑用户可能的背景。可能是普通用户，没有特别专业的背景，所以需要用简单易懂的语言，避免使用专业术语。同时，用户可能希望得到结构清晰的回答，分点列出，方便阅读。\n\n然后，我需要确保涵盖健康的主要方面。通常，健康建议包括均衡饮食、规律运动、充足睡眠、心理健康、避免不良习
惯、定期体检等。每个点需要简明扼要，给出具体例子，比如饮食方面提到多吃蔬菜水果，少油少盐；运动方面建议每周至少150分钟中等强度运动。\n\n还要注意用户可能没有提到的深层需求，比如如何坚持健康习惯，或者如何处理压力。这时候可以加入一些实用的小贴士，比如设定小目标、找到运动伙伴、保持积极心态等。\n\n另外，用户要求用中文回答，所以整个思考过程需要用中文，但这里只是模拟内部思考，所以可以先用英文思考，再翻译成中文。需要确保回答流畅自然，符合中文表达习惯。\n\n最后，检查是否有遗漏的重要点，比如心理健
康的重要性，或者避免吸烟酗酒等不良习惯。确保信息全面，同时不过于冗长。可能还需要提醒用户如果有特殊情况应咨询专业人士，但用户没有提到，所以可能不需要深入，但可以稍微提及。\n\n总结下来，回答的结构应该是分点列出，每点下有简短的建议，使用清晰的标题，如饮食、运动、睡眠等，每个部分用一两句话说明，最后可以加一个总结句，鼓励用户坚持。这样既全面又易于阅读。 [unused17] 保持健康需要从多个方面入手，以下是一些实用建议：\n\n### 1. **均衡饮食**\n   - **多样化摄入**：多吃蔬菜
、水果、全谷物、优质蛋白（如鱼、豆类、瘦肉），减少高糖、高盐、高脂肪食物。\n   - **控制分量**：避免暴饮暴食，七八分饱即可。\n   - **多喝水**：每天至少喝1.5-2升水，少喝含糖饮料。\n\n### 2. **规律运动**\n   - **每周至少150分钟**：如快走、游泳、骑自行车等中等强度运动。\n   - **力量训练**：每周2次，增强肌肉和骨骼健康。\n\n### 3. **充足","tool_calls":[]},"logprobs":null,"finish_rea
son":"length","stop_reason":null}],"usage":{"prompt_tokens":21,"total_tokens":533,"completion_tokens":512,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_transfer_params":null}
```
### 1B模型推理

- kill掉7b推理进程，然后编写拉起1b推理脚本: run_service_1b.sh

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
HOST=192.168.0.233    # 根据实际情况进行修改
PORT=1040             # 根据实际情况进行修改
LOCAL_CKPT_DIR=/home/Pangu/openPangu-Embedded-1B-V1.1
SERVED_MODEL_NAME=pangu_embedded_1b
vllm serve $LOCAL_CKPT_DIR \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --max-num-seqs 16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 2048 \
    --tokenizer-mode "slow" \
    --dtype bfloat16 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
```

- 拉起服务脚本

```bash
nohup bash run_service_1b.sh > run_1b.log 2>&1 &
```

- 运行如下测试脚本：

```bash
MASTER_NODE_IP=192.168.0.233  # server node ip
PORT=1040
SERVED_MODEL_NAME=pangu_embedded_1b
curl http://${MASTER_NODE_IP}:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$SERVED_MODEL_NAME'",
        "messages": [
            {
                "role": "user",
                "content": "How to keep healthy. Response in Chinese!"
            }
        ],
        "max_tokens": 512,
        "temperature": 0
    }'
```

测试结果：
```
{"id":"chatcmpl-4bf64f945d4e46288189afce0b008fbd","object":"chat.completion","created":1758704116,"model":"pangu_embedded_1b","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"保持健康需要从生活习惯、饮食、运动和心态等多方面入手。以下是一些实用的建议，帮
助你养成健康的生活方式：\n\n---\n\n### **1. 均衡饮食**\n- **多样化摄入**：多吃蔬菜、水果、全谷物、优质蛋白（如鱼、豆类、瘦肉）和健康脂肪（如坚果、橄榄油）。\n- **减少加工食品**：避免高糖、高盐、高脂肪的零食和快餐，选择天然食材。\n- **控制分量**：细嚼慢咽，避免暴饮暴食，注意饮食节奏。\n\n---\n\n### **2. 规律运动**\n- **有氧运动**：每周至少150分钟快走、游泳或骑自行车，帮助燃烧脂肪。\n- **力量训练**：每周2-3次力量练
习（如深蹲、俯卧撑），增强肌肉和骨骼。\n- **日常活动**：多走路、爬楼梯，避免久坐，每小时起身活动5分钟。\n\n---\n\n### **3. 充足睡眠**\n- 成年人每天保证7-9小时睡眠，避免熬夜。\n- 睡前减少屏幕时间，保持卧室黑暗、安静。\n\n---\n\n### **4. 管理压力**\n- 通过冥想、深呼吸、瑜伽或兴趣爱好缓解压力。\n- 保证充足休息，避免过度劳累。\n\n---\n\n### **5. 定期体检**\n- 每年检查一次血压、血糖、胆固醇等指标，早期发现潜在
问题。\n\n---\n\n### **6. 戒烟限酒**\n- 吸烟会损害呼吸系统和心血管健康，尽量戒烟。\n- 饮酒适量（男性≤2杯/天，女性≤1杯/天），避免过量。\n\n---\n\n### **7. 保持社交与兴趣**\n- 与家人朋友保持联系，避免孤独。\n- 培养兴趣爱好（如阅读、绘画、运动），提升生活幸福感。\n\n---\n\n### **8. 其他小贴士**\n- **多喝水**：每天喝够1.5-2升水，避免含糖饮料。\n- **减少屏幕时间**：长时间看手机/电脑可能影响视力、颈
椎和睡眠。\n- **适度运动**：即使每天散步30分钟，也能显著提升健康水平。\n\n---\n\n健康是长期积累的结果，从小习惯开始，逐步调整，你会感受到身体和精神的改善！ �","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":20,"total_tokens":465,"completion_tokens":445,"prompt_token
s_details":null},"prompt_logprobs":null,"kv_transfer_params":null}
```
