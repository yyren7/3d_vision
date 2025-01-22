import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from vllm import LLM, SamplingParams

model = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    quantization="awq",  # 使用AWQ量化方法
    gpu_memory_utilization=0.85,  # 降低显存利用率阈值
)

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.5,
    top_p=0.9,
    max_tokens=256,  # 减少生成的最大token数
    skip_special_tokens=True  # 跳过特殊token减少处理
)

# 输入提示（根据DeepSeek的推荐格式）
prompt = """<｜begin▁of▁sentence｜>Hi, can you explain quantum computing in simple terms?<｜end▁of▁sentence｜>"""

# 运行推理
try:
    outputs = model.generate([prompt], sampling_params)
finally:
    # 显式释放模型资源
    del model
    import torch
    torch.cuda.empty_cache()

# 输出结果
# 流式处理减少峰值内存
for output in model.generate_stream([prompt], sampling_params):
    print(output.outputs[0].text, end="", flush=True)