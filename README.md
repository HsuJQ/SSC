# README

## 安装eai测评工具

1. **创建激活conda环境**:
   ```bash
   conda create -n eai-eval python=3.8 -y 
   conda activate eai-eval
   ```

2. **安装 `eai`**:
   
   通过pip安装:
   ```bash
   pip install eai-eval
   ```
    或者通过源安装
   ```bash
   git clone https://github.com/embodied-agent-interface/embodied-agent-interface.git
   cd embodied-agent-interface
   pip install -e .
   ```

3. **（可选）测试 PDDL 规划器以进行 transition modeling**:
    ```bash
    python examples/pddl_tester.py
    ```
    如果输出结果为 `Results: ['walk_towards character light', 'switch_on character light']`，则安装成功。否则，参考 `pddlgym_planners/` 目录下的 `BUILD.md` 文件。

## 运行pangu模型并测试（由于晟腾已预配置pangu，故不再创建环境和下载模型）

0. 若无盘古模型，请按照下述方法安装
   
   1. 切换到指定文件夹
   ```bash
   cd /opt/pangu/
   ```
   2. 下载
   ```bash
   git clone https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B-V1.1
   ```
1. 切换到指定文件夹
   
```bash
cd /opt/pangu/examples/vllm-inference
```
   
2. 激活虚拟环境

```bash
conda activate pangu
```

3. 运行pangu模型

```bash
# 在前四张卡上跑
bash run_service_7b_0123.sh

# 在后四张卡上跑
bash run_service_7b_4567.sh
```

4. 利用pangu模型生成输出

```python
# 利用0123上的模型进行输出
python client_generate_0123.py

# 利用4567上的模型进行输出
python client_generate_4567.py
```

5. 切换虚拟环境

```bash
conda activate eai-eval
```

6. 测试

```bash
eai-eval --dataset virtualhome --eval-type action_sequencing --mode evaluate_results --llm-response-path /opt/pangu/examples/vllm-inference/virtualhome/test/llm_output
eai-eval --dataset virtualhome --eval-type goal_interpretation --mode evaluate_results --llm-response-path /opt/pangu/examples/vllm-inference/virtualhome/test/llm_output
eai-eval --dataset virtualhome --eval-type subgoal_decomposition --mode evaluate_results --llm-response-path /opt/pangu/examples/vllm-inference/virtualhome_m/test/llm_output
eai-eval --dataset virtualhome --eval-type transition_modeling --mode evaluate_results --llm-response-path /opt/pangu/examples/vllm-inference/virtualhome_m/test/llm_output
```
## 相关文件解释

1. 调用大模型生成结果的日志为：`client_generate_0123.log`,`client_generate_4567.log`

2. 测试结果的日志为：`/opt/pangu/examples/vllm-inference/virtualhome/test/logs/goal_interpretation_eval_20251212_154941.log`,`/opt/pangu/examples/vllm-inference/virtualhome/test/logs/action_sequencing_eval_20251212_160842.log`,`/opt/pangu/examples/vllm-inference/virtualhome_m/test/logs/subgoal_decomposition_eval_20251212_154908.log`,`/opt/pangu/examples/vllm-inference/virtualhome_m/test/logs/transition_model_eval_20251212_154917.log`

3. 测试结果为：`/opt/pangu/examples/vllm-inference/virtualhome/test/output/virtualhome/evaluate_results/goal_interpretation`,`/opt/pangu/examples/vllm-inference/virtualhome/test/output/virtualhome/evaluate_results/action_sequencing`,`/opt/pangu/examples/vllm-inference/virtualhome_m/test/output/virtualhome/evaluate_results/subgoal_decomposition`,`/opt/pangu/examples/vllm-inference/virtualhome_m/test/output/virtualhome/evaluate_results/transition_modeling`
