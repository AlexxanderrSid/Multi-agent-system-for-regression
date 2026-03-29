import re
import pandas as pd


_FORBIDDEN = [
    r'os\.system\(',
    r'subprocess\.',
    r'__import__\(',
    r"open\([^)]*['\"][wWaA]",
    r'shutil\.rmtree',
]

_LEAKAGE = [
    r'scaler\.fit\s*\(.*test',
    r'encoder\.fit\s*\(.*test',
    r'fit_transform\s*\(.*test',
]


def validate_input_data(df) -> tuple:
    """Проверка корректности входных данных."""
    issues = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        issues.append('DataFrame пустой или некорректный')
    if 'target' not in df.columns:
        issues.append('Не найдена колонка target')
    if len(df) < 10:
        issues.append(f'Слишком мало строк: {len(df)}')
    if 'target' in df.columns and not pd.api.types.is_numeric_dtype(df['target']):
        issues.append('target должен быть числовым для регрессии')
    return len(issues) == 0, issues


def check_code_safety(code: str) -> tuple:
    """Проверка безопасности сгенерированного кода."""
    violations = [p for p in _FORBIDDEN if re.search(p, code)]
    return len(violations) == 0, violations


def check_data_leakage(code: str) -> tuple:
    """Предупреждение о возможной утечке данных."""
    found = [p for p in _LEAKAGE if re.search(p, code, re.IGNORECASE)]
    return len(found) == 0, found


def sanitize_code(text: str) -> str:
    """Извлечь Python-код из ответа модели."""
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return '\n\n'.join(b.strip() for b in blocks)
    blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return '\n\n'.join(b.strip() for b in blocks)
    return text.strip()
