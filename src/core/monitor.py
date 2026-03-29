import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger('MAS')


@dataclass
class Record:
    ts: str
    agent: str
    action: str
    score: Optional[float] = None
    ok: bool = True


class Monitor:
    """Логирование всех событий системы."""

    def __init__(self):
        self.records: List[Record] = []
        self.best_score = float('inf')
        self.best_label = ''
        self.llm_calls = 0
        self.exec_calls = 0
        self.debug_calls = 0
        self.t0 = time.time()

    def log(self, agent: str, action: str,
            score: Optional[float] = None, ok: bool = True):
        r = Record(datetime.now().strftime('%H:%M:%S'), agent, action, score, ok)
        self.records.append(r)
        icon = '✅' if ok else '-'
        s = f' | MSE={score:.4f}' if score is not None else ''
        logger.info(f'{icon} [{agent}] {action}{s}')
        if score is not None and score < self.best_score:
            self.best_score = score
            self.best_label = f'{agent}::{action}'

    def report(self) -> str:
        elapsed = time.time() - self.t0
        rows = [r for r in self.records if r.score is not None]
        lines = [
            '\n' + '='*55,
            'ИТОГОВЫЙ ОТЧЁТ',
            '='*55,
            f'Время: {elapsed:.0f}с',
            f'LLM вызовов: {self.llm_calls}',
            f'Выполнений кода: {self.exec_calls}',
            f'Debug попыток: {self.debug_calls}',
            f'Лучший MSE: {self.best_score:.4f}  [{self.best_label}]',
            '',
            'История score (MSE):',
        ]
        for r in rows:
            flag = '✅' if r.score == self.best_score else '  '
            lines.append(f'  {flag} {r.ts} [{r.agent}] {r.action}: {r.score:.4f}')
        lines.append('='*55)
        return '\n'.join(lines)


MON = Monitor()
