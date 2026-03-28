import json
import re
from typing import List, Tuple

from src.core.llm_client import LLMClient


class AblationAgent:
    """
    Проводит ablation study:
    - Генерирует код, отключающий отдельные компоненты pipeline
    - Суммаризирует результаты
    - Предлагает следующий целевой компонент для улучшения
    - Хранит память предыдущих ablation, чтобы не повторяться
    """

    SYS_ABLATION = (
        'You are a Kaggle expert conducting an ablation study.\n'
        'Output ONLY Python code in ```python``` block.\n'
        'Use df variable. Test 3 variations of the pipeline.\n'
        "For each print: 'Ablation {name}: {score:.4f}'\n"
        "Print at end: 'Most impactful: {component}'"
    )

    SYS_PLAN = 'You are a Kaggle expert planning improvements for regression (minimize MSE).'

    def __init__(self, llm: LLMClient):
        self.llm    = llm
        self.name   = 'AblationAgent'
        self.memory: List[str] = []

    def run(self, code_snippet: str, info: str) -> str:
        already = ', '.join(self.memory[-3:]) if self.memory else 'none'
        prompt = (
            f'Current solution snippet:\n```python\n{code_snippet[:2000]}\n```\n'
            f'Dataset info: {info[:300]}\n'
            f'Already studied: {already}\n\n'
            'Generate ablation study code that tests 3 variations:\n'
            '- Variation 1: disable a feature engineering step\n'
            '- Variation 2: change preprocessing (imputation strategy)\n'
            '- Variation 3: change model hyperparameters\n'
            "Print 'Most impactful: {component}' at the end."
        )
        return self.llm.call(self.SYS_ABLATION, prompt, max_tokens=1100)

    def summarize(self, abl_code: str, abl_output: str) -> str:
        prompt = (
            f'Ablation output:\n{abl_output[:800]}\n\n'
            'Summarize in 3 sentences: which component matters most? '
            'What should be improved next?'
        )
        summary = self.llm.call(
            'Summarize ablation results concisely.',
            prompt, max_tokens=250
        ).strip()
        self.memory.append(summary[:80])
        return summary

    def propose_plan(self, summary: str,
                     tried_plans: List[str]) -> Tuple[str, str]:
        tried = '\n'.join(f'- {p}' for p in tried_plans[-3:]) or 'none'
        prompt = (
            f'Ablation summary:\n{summary}\n\n'
            f'Already tried plans:\n{tried}\n\n'
            'Propose the NEXT improvement plan (3-4 sentences, specific).\n'
            'Also name the target component.\n'
            'Reply as JSON: {"plan": "...", "component": "..."}'
        )
        resp = self.llm.call(self.SYS_PLAN, prompt, max_tokens=350)
        try:
            m = re.search(r'\{.*?\}', resp, re.DOTALL)
            if m:
                d = json.loads(m.group())
                return d.get('plan', resp), d.get('component', 'general')
        except Exception:
            pass
        return resp.strip(), 'general'
