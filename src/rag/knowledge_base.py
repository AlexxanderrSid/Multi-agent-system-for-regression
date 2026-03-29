from typing import List, Dict


_KB_DOCS = [
    {
        'id': 'preprocessing',
        'topic': 'preprocessing cleaning',
        'text': (
            'Data preprocessing rules:\n'
            '- NEVER fit scaler/encoder on test data, only transform\n'
            '- Fill numeric NaN with median computed on train\n'
            '- Fill categorical NaN with mode or special value MISSING\n'
            '- Convert datetime columns: extract year, month, day, dayofweek, days_since\n'
            '- Drop id/name text columns (low signal, high cardinality)\n'
            '- Drop duplicate rows'
        )
    },
    {
        'id': 'feature_eng',
        'topic': 'feature engineering airbnb',
        'text': (
            'Effective features for Airbnb-like regression:\n'
            '- log1p(price/sum) — log-transform skewed prices\n'
            '- days_since_last_review = (today - last_dt).days\n'
            '- reviews_per_day = amt_reviews / max(days_active, 1)\n'
            '- price_per_min_day = sum / min_days\n'
            '- is_popular_host = (total_host > 3).astype(int)\n'
            '- LabelEncode: type_house, location_cluster, location\n'
            '- GroupBy mean price per location_cluster'
        )
    },
    {
        'id': 'models',
        'topic': 'models regression',
        'text': (
            'Best models for tabular regression (MSE):\n'
            '1. LightGBM - fast, handles NaN natively, top performer\n'
            '2. XGBoost - stable, good defaults\n'
            '3. CatBoost - great for categoricals\n'
            '4. RandomForest - robust baseline\n'
            'Use random 80/20 split. Metric: mean_squared_error (lower is better).\n'
            'Store model as best_model, features as feature_cols, score as val_score (MSE).'
        )
    },
    {
        'id': 'ensemble',
        'topic': 'ensemble stacking',
        'text': (
            'Ensemble strategies for regression:\n'
            '1. Simple average of predictions\n'
            '2. Weighted average (weights = 1/val_mse, normalised)\n'
            '3. Stacking: Ridge meta-learner on OOF predictions\n'
            '4. Rank averaging: average rank-normalised predictions'
        )
    },
]


class KnowledgeBase:
    """RAG-модуль: простой keyword-based retrieval."""

    def __init__(self, docs: List[Dict]):
        self.docs = docs

    def retrieve(self, query: str, top_k: int = 2) -> str:
        q = query.lower()
        scored = []
        for d in self.docs:
            score = sum(1 for w in q.split() if w in d['topic'])
            scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        top = [d for _, d in scored[:top_k]]
        if not top:
            top = self.docs[:top_k]
        return '\n---\n'.join(d['text'] for d in top)


KB = KnowledgeBase(_KB_DOCS)
