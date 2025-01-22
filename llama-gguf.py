from llama_cpp import Llama

# 初始化模型
llm = Llama(
    model_path="../Downloads/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",  # 替换为你的 GGUF 文件路径
    n_ctx=2048,          # 上下文长度（需与模型训练时对齐）
    n_threads=8,         # CPU 线程数（物理核心数，非逻辑线程）
    n_gpu_layers=35,     # 启用 GPU 加速的层数（设为 0 表示仅用 CPU）
    verbose=True         # 显示加载过程日志
)

# 执行推理
prompt = "What is the capital of France?"
response = llm(
    prompt=prompt,
    max_tokens=100,      # 生成的最大 token 数
    temperature=0.8,     # 随机性控制（0=确定性，1=随机）
    stop=["\n", "###"]   # 停止生成的条件
)

# 输出结果
print(response["choices"][0]["text"])