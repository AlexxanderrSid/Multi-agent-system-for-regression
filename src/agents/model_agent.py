from src.core.llm_client import LLMClient
from src.rag.knowledge_base import KnowledgeBase


class ModelAgent:
    """
    Обучает ML-модели.
    - Генерирует код начального обучения (LightGBM/XGBoost)
    - Рефайнмент: улучшает существующий код по плану
    - Хранит реестр обученных моделей для ансамбля
    """

    SYS = (
        'You are a Kaggle grandmaster for regression tasks.\n'
        'Output ONLY executable Python code in ```python``` block.\n'
        'Use df variable (preprocessed DataFrame with target column).\n'
        'Random 80/20 split, random_state=42.\n'
        'Metric: mean_squared_error on val set (lower is better).\n'
        "Print EXACTLY: 'Final Validation Performance: {val:.4f}'\n"
        'Store: best_model, feature_cols (list), val_score (float MSE), '
        'label_encoders (dict, empty if not used).'
    )

    def __init__(self, llm: LLMClient, kb: KnowledgeBase):
        self.llm = llm
        self.kb = kb
        self.name = 'ModelAgent'

    def run(self, info: str, model_name: str = 'LightGBM') -> str:
        rag = self.kb.retrieve('models regression')
        prompt = (
            f'Dataset info:\n{info}\n\n'
            f'Knowledge:\n{rag}\n\n'
            f'Train {model_name} regressor:\n'
            '1. Drop non-numeric / encode categoricals with LabelEncoder.\n'
            '2. Fill remaining NaN with median.\n'
            '3. Random 80/20 split.\n'
            f'4. Fit {model_name} regressor on train split only.\n'
            '5. Compute mean_squared_error on val split.\n'
            "6. Print 'Final Validation Performance: {val:.4f}'\n"
            '7. Store best_model, feature_cols, val_score (MSE), label_encoders.'
        )
        return self.llm.call(self.SYS, prompt, max_tokens=1100)

    def run_refined(self, cur_code: str, plan: str, cur_score: float) -> str:
        prompt = (
            f'Current code (val MSE={cur_score:.4f}):\n'
            f'```python\n{cur_code[:2500]}\n```\n\n'
            f'Improvement plan:\n{plan}\n\n'
            'Implement the plan. Keep same output format.\n'
            "Print 'Final Validation Performance: {val:.4f}'\n"
            'Store best_model, feature_cols, val_score (MSE), label_encoders.'
        )
        return self.llm.call(self.SYS, prompt, max_tokens=1300)
