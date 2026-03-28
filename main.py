import os
import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.core.llm_client import LLMClient
from src.rag.knowledge_base import KB
from src.system import MultiAgentMLSystem
from src.benchmark import run_benchmark, compare


MODEL_PATH = hf_hub_download(
    repo_id='Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
    filename='qwen2.5-coder-7b-instruct-q4_k_m.gguf',
    local_dir='./models'
)

llm_model = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    n_batch=512,
    verbose=False
)

LLM = LLMClient(llm_model)


# ── Загрузка данных ──────────────────────────────────────────
# Вариант 1: Kaggle
# df_train = pd.read_csv('/kaggle/input/mws-ai-agents-2026/train.csv')
# df_test  = pd.read_csv('/kaggle/input/mws-ai-agents-2026/test.csv')

# Вариант 2: локально / Colab
df_train = pd.read_csv('data/train.csv')
df_test  = pd.read_csv('data/test.csv')


bench = run_benchmark(df_train)

system = MultiAgentMLSystem(df_train=df_train, df_test=df_test, llm=LLM, kb=KB)
final_score = system.run()

comparison = compare(bench, final_score)
