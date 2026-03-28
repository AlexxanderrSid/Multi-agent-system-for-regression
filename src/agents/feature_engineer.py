from typing import Dict, List

from src.core.llm_client import LLMClient
from src.rag.knowledge_base import KnowledgeBase


class FeatureEngineerAgent:
    """
    Создаёт новые признаки.
    - RAG: лучшие практики feature engineering
    - Plan trajectory: не повторяет уже пробованные планы
    - Поддерживает рефайнмент по новому плану
    """

    SYS = (
        'You are a feature engineering expert for Kaggle binary classification.\n'
        'Output ONLY executable Python code inside ```python``` block.\n'
        'Use df (train) and df_test (test, no target col).\n'
        'Apply same transforms to df_test.\n'
        "Print 'FE done. New cols: N'"
    )

    def __init__(self, llm: LLMClient, kb: KnowledgeBase):
        self.llm  = llm
        self.kb   = kb
        self.name = 'FeatureEngineerAgent'
        self.plan_trajectory: List[Dict] = []

    def run(self, info: str, ablation_hint: str = '') -> str:
        rag = self.kb.retrieve('feature engineering airbnb')

        history = ''
        if self.plan_trajectory:
            history = 'Already tried (do NOT repeat):\n' + '\n'.join(
                f'- {p["plan"]}' for p in self.plan_trajectory[-3:]
            )

        hint = f'Ablation hint: {ablation_hint}\n' if ablation_hint else ''

        prompt = (
            f'Dataset columns: {info}\n\n'
            f'Knowledge:\n{rag}\n\n'
            f'{hint}'
            f'{history}\n\n'
            'Create 5-8 new features:\n'
            '- Date features (days_since_last_review from last_dt)\n'
            '- Log-transform price (sum)\n'
            '- LabelEncode categoricals\n'
            '- Ratio features (amt_reviews/min_days, etc.)\n'
            'Apply same to df_test.\n'
            "Print 'FE done. New cols: N'"
        )
        return self.llm.call(self.SYS, prompt, max_tokens=1000)

    def propose_new_plan(self, cols: List[str], tried: List[Dict]) -> str:
        """Предложить следующий план FE (для inner loop Planner)."""
        tried_str = '\n'.join(
            f'- {p["plan"]} → {p.get("score","?"):.4f}'
            for p in tried[-3:]
        ) or 'None'
        prompt = (
            f'Columns: {cols}\n'
            f'Already tried:\n{tried_str}\n\n'
            'Suggest ONE new feature engineering plan (2-4 sentences).\n'
            'Be specific. No code, just the plan description.'
        )
        return self.llm.call(
            'You are a Kaggle feature engineering expert for regression.',
            prompt, max_tokens=300
        ).strip()
