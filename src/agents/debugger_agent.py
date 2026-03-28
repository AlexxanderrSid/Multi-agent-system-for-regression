from src.core.llm_client import LLMClient


class DebuggerAgent:
    """Автоматическое исправление ошибок в сгенерированном коде."""

    SYS = (
        'You are a Python debugging expert.\n'
        'Fix the code. Output ONLY the corrected code in ```python``` block.\n'
        'Keep same logic and print statements.'
    )

    def __init__(self, llm: LLMClient):
        self.llm  = llm
        self.name = 'DebuggerAgent'

    def fix(self, code: str, error: str) -> str:
        prompt = (
            f'Code:\n```python\n{code[:2500]}\n```\n\n'
            f'Error:\n{error[:600]}\n\n'
            'Fix the error. Return corrected ```python``` code only.'
        )
        return self.llm.call(self.SYS, prompt, max_tokens=1300)
