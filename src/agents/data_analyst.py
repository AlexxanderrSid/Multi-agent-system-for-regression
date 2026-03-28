from src.core.llm_client import LLMClient
from src.rag.knowledge_base import KnowledgeBase


class DataAnalystAgent:
    """
    Анализирует датасет и генерирует код очистки.
    - Использует RAG (preprocessing rules)
    - Применяет трансформации к df и df_test одинаково
    - Обеспечивает: нет fit на test, нет утечки
    """

    SYS = (
        'You are a data cleaning expert for Kaggle competitions.\n'
        'Output ONLY executable Python code inside ```python``` block.\n'
        'Variable df = train DataFrame (already loaded). '
        'Variable df_test = test DataFrame (may be None).\n'
        'NEVER fit any transformer on df_test.\n'
        "Print 'Cleaning done. Shape: {df.shape}' at the end."
    )

    def __init__(self, llm: LLMClient, kb: KnowledgeBase):
        self.llm = llm
        self.kb  = kb
        self.name = 'DataAnalystAgent'

    def run(self, info: str) -> str:
        rag = self.kb.retrieve('preprocessing cleaning')
        prompt = (
            f'Dataset info:\n{info}\n\n'
            f'Knowledge:\n{rag}\n\n'
            'Tasks:\n'
            '1. Drop useless text cols (host_name, name, _id if not needed).\n'
            '2. Fill NaN: numeric→median(train), categorical→mode(train).\n'
            '3. Convert last_dt / date cols to days_since (int).\n'
            '4. Drop duplicates.\n'
            'Apply same transforms to df_test (no fit on test).\n'
            "Print 'Cleaning done. Shape: {df.shape}'"
        )
        return self.llm.call(self.SYS, prompt, max_tokens=900)
