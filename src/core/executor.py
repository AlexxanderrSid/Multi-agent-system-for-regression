import re
import traceback
import logging
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Tuple

from src.core.guardrails import check_code_safety, check_data_leakage
from src.core.monitor import MON

logger = logging.getLogger('MAS')


def safe_execute(code: str,
                 ctx: Dict[str, Any]) -> Tuple[bool, str, Dict]:
    """
    Выполнить код с:
    - проверкой безопасности (guardrails)
    - захватом stdout
    - возвратом обновлённого контекста
    """
    ok, violations = check_code_safety(code)
    if not ok:
        return False, f'SECURITY: {violations}', ctx

    clean, warns = check_data_leakage(code)
    if not clean:
        logger.warning(f'⚠️ Possible data leakage: {warns}')

    if not code.strip():
        return False, 'Empty code', ctx

    buf = StringIO()
    local = ctx.copy()
    MON.exec_calls += 1
    try:
        with redirect_stdout(buf):
            exec(code, local)  # noqa: S102
        return True, buf.getvalue(), local
    except Exception:
        MON.debug_calls += 1
        return False, traceback.format_exc(), ctx


def extract_score(text: str) -> Optional[float]:
    """Найти последнее число в тексте вывода — это MSE (любое положительное число)."""
    m = re.search(r'Final Validation Performance:\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)', text)
    if m:
        return float(m.group(1))
    nums = re.findall(r'[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?', text)
    for n in reversed(nums):
        v = float(n)
        if v > 0:
            return v
    return None
