from typing import Dict, List

from src.core.llm_client import LLMClient
from src.rag.knowledge_base import KnowledgeBase


class EnsembleAgent:
    """
    Итеративно предлагает и реализует стратегии ансамблирования.
    Хранит историю попыток для улучшения планов (plan trajectory).
    """

    SYS = (
        'You are a Kaggle grandmaster for ensemble regression methods.\n'
        'Output ONLY Python code in ```python``` block.\n'
        'Variable models_registry is a list of dicts with keys:\n'
        '  model, feature_cols, val_score (MSE), label_encoders.\n'
        'Use df for train data. Evaluate on random 80/20 val split.\n'
        'Metric: mean_squared_error (lower is better).\n'
        "Print 'Final Validation Performance: {val:.4f}'"
    )

    def __init__(self, llm: LLMClient, kb: KnowledgeBase):
        self.llm = llm
        self.kb = kb
        self.name = 'EnsembleAgent'
        self.history: List[Dict] = []

    def propose_plan(self, models_info: str) -> str:
        rag = self.kb.retrieve('ensemble stacking')
        tried = '\n'.join(
            f'- {h["plan"]} → MSE={h.get("score","?"):.4f}'
            for h in self.history[-3:]
        ) or 'none'
        prompt = (
            f'Models:\n{models_info}\n\n'
            f'Ensemble knowledge:\n{rag}\n\n'
            f'Already tried:\n{tried}\n\n'
            'Propose a NEW ensemble strategy for regression (2-3 sentences). No code.'
        )
        return self.llm.call(
            'Kaggle ensemble expert for regression.', prompt, max_tokens=300
        ).strip()

    def implement_plan(self, plan: str, models_info: str) -> str:
        prompt = (
            f'Models in models_registry:\n{models_info}\n\n'
            f'Ensemble plan:\n{plan}\n\n'
            'Implement this ensemble. Use models from models_registry.\n'
            'Random 80/20 split, random_state=42.\n'
            'Metric: mean_squared_error (lower is better).\n'
            "Print 'Final Validation Performance: {val:.4f}'"
        )
        return self.llm.call(self.SYS, prompt, max_tokens=1100)
