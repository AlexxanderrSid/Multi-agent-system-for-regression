from llama_cpp import Llama
from src.core.monitor import MON


class LLMClient:
    """
    Обёртка над llama_cpp.Llama.
    Использует Chat Completion API (system / user роли).
    Модель: Qwen2.5-Coder-7B-Instruct-Q4_K_M (open-source, Apache 2.0).
    """

    def __init__(self, model: Llama):
        self._m = model

    def call(self, system: str, user: str,
             max_tokens: int = 1024,
             temperature: float = 0.2) -> str:
        MON.llm_calls += 1
        resp = self._m.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=['```\n\n', '<|im_end|>'],
        )
        return resp['choices'][0]['message']['content']
