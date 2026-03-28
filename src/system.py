import logging
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import OUTER_LOOP_STEPS, INNER_LOOP_STEPS, MAX_DEBUG, ENSEMBLE_ROUNDS
from src.core.guardrails import validate_input_data, sanitize_code
from src.core.executor import safe_execute, extract_score
from src.core.monitor import MON
from src.core.llm_client import LLMClient
from src.rag.knowledge_base import KnowledgeBase
from src.agents.data_analyst import DataAnalystAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_agent import ModelAgent
from src.agents.ablation_agent import AblationAgent
from src.agents.ensemble_agent import EnsembleAgent
from src.agents.debugger_agent import DebuggerAgent

logger = logging.getLogger('MAS')


class MultiAgentMLSystem:
    """
    Мультиагентная система регрессии.
    Архитектура по MLE-STAR: search → initial solution →
    outer loop (ablation → target component) →
    inner loop (planner → coder → evaluate) → ensemble.
    Метрика: MSE (чем меньше — тем лучше).
    """

    def __init__(self, df_train: pd.DataFrame,
                 df_test: pd.DataFrame = None,
                 llm: LLMClient = None,
                 kb: KnowledgeBase = None):
        ok, issues = validate_input_data(df_train)
        if not ok:
            raise ValueError(f'Input validation failed: {issues}')

        self.ctx: Dict[str, Any] = {
            'df':      df_train.copy(),
            'df_test': df_test.copy() if df_test is not None else None,
            'np':      np,
            'pd':      pd,
        }

        self.data_agent     = DataAnalystAgent(llm, kb)
        self.feat_agent     = FeatureEngineerAgent(llm, kb)
        self.model_agent    = ModelAgent(llm, kb)
        self.ablation_agent = AblationAgent(llm)
        self.ensemble_agent = EnsembleAgent(llm, kb)
        self.debugger       = DebuggerAgent(llm)

        self.best_score     = float('inf')
        self.best_code      = ''
        self.models_reg: List[Dict] = []
        self.tried_plans:  List[str] = []

        logger.info('System initialised. Train=%s Test=%s',
                    df_train.shape,
                    df_test.shape if df_test is not None else None)

    def _info(self) -> str:
        df = self.ctx['df']
        return (
            f'Shape: {df.shape}\n'
            f'Columns: {list(df.columns)}\n'
            f'Dtypes: {dict(df.dtypes.astype(str))}\n'
            f'NaN: {df.isnull().sum().to_dict()}\n'
            f'Target stats: min={df["target"].min():.2f}, '
            f'max={df["target"].max():.2f}, '
            f'mean={df["target"].mean():.2f}, '
            f'std={df["target"].std():.2f}'
        )

    def _run(self, code: str, tag: str) -> Tuple[bool, str]:
        """Выполнить код (с авто-дебагом до MAX_DEBUG попыток)."""
        cur = sanitize_code(code)
        for attempt in range(MAX_DEBUG):
            ok, out, new_ctx = safe_execute(cur, self.ctx)
            if ok:
                self.ctx.update(new_ctx)
                if attempt > 0:
                    logger.info('Fixed after %d attempt(s): %s', attempt, tag)
                return True, out
            if attempt < MAX_DEBUG - 1:
                logger.warning('Debug attempt %d/%d [%s]', attempt+1, MAX_DEBUG, tag)
                fixed = self.debugger.fix(cur, out)
                cur = sanitize_code(fixed)
        MON.log(tag, 'failed', ok=False)
        return False, out

    def _save_model(self, score: float):
        if 'best_model' in self.ctx:
            self.models_reg.append({
                'model':          self.ctx['best_model'],
                'feature_cols':   self.ctx.get('feature_cols', []),
                'val_score':      score,
                'label_encoders': self.ctx.get('label_encoders', {}),
            })

    def step_clean(self):
        print('\n' + '='*50)
        print('📊 ШАГ 1: Очистка данных')
        print('='*50)
        code = self.data_agent.run(self._info())
        ok, out = self._run(code, 'DataClean')
        print('✅' if ok else '⚠️ ошибка (продолжаем)', '|', out[:120])
        MON.log(self.data_agent.name, 'clean', ok=ok)

    def step_features(self, hint: str = '') -> str:
        print('\n' + '='*50)
        print('🔧 ШАГ 2: Feature Engineering')
        print('='*50)
        cols = str(list(self.ctx['df'].columns))
        code = self.feat_agent.run(cols, hint)
        ok, out = self._run(code, 'FeatEng')
        print('✅' if ok else '⚠️ ошибка', '|', out[:120])
        MON.log(self.feat_agent.name, 'features', ok=ok)
        return sanitize_code(code)

    def step_train(self, model_name: str = 'LightGBM') -> Tuple[float, str]:
        print('\n' + '='*50)
        print(f'🤖 ШАГ 3: Обучение ({model_name})')
        print('='*50)
        code = self.model_agent.run(self._info(), model_name)
        ok, out = self._run(code, f'Train_{model_name}')
        score = extract_score(out) if ok else None
        if score is not None and score > 0:
            print(f'  MSE = {score:.4f}')
            MON.log(self.model_agent.name, f'train_{model_name}', score=score)
            if score < self.best_score:
                self.best_score = score
                self.best_code  = sanitize_code(code)
            self._save_model(score)
        else:
            print('  ⚠️ Не удалось получить score:', out[:200])
        return score or float('inf'), sanitize_code(code)

    def step_refine(self, cur_code: str) -> str:
        print('\n' + '='*50)
        print('🔬 ШАГ 4: Ablation + Refinement')
        print('='*50)

        info = self._info()

        for outer in range(OUTER_LOOP_STEPS):
            print(f'\n  ── Outer {outer+1}/{OUTER_LOOP_STEPS} ──')

            abl_code = self.ablation_agent.run(cur_code, info)
            ok, abl_out = self._run(abl_code, f'Ablation_o{outer}')
            if not ok:
                abl_out = 'Ablation failed'

            summary = self.ablation_agent.summarize(abl_code, abl_out)
            print(f'     Summary: {summary[:130]}')

            plan, comp = self.ablation_agent.propose_plan(summary, self.tried_plans)
            print(f'     Target: {comp} | Plan: {plan[:100]}')

            best_inner = self.best_score
            best_code_inner = cur_code

            for inner in range(INNER_LOOP_STEPS):
                print(f'       Inner {inner+1}/{INNER_LOOP_STEPS}: {plan[:70]}')

                ref_code = self.model_agent.run_refined(cur_code, plan, self.best_score)
                ok, out = self._run(ref_code, f'Refine_o{outer}_i{inner}')

                if ok:
                    sc = extract_score(out)
                    if sc is not None:
                        print(f'         → MSE={sc:.4f} (best {best_inner:.4f})')
                        MON.log('Refine', f'o{outer}_i{inner}', score=sc)
                        if sc < best_inner:
                            best_inner = sc
                            best_code_inner = sanitize_code(ref_code)

                if inner < INNER_LOOP_STEPS - 1:
                    plan = self.feat_agent.propose_new_plan(
                        list(self.ctx['df'].columns),
                        [{'plan': plan, 'score': best_inner}]
                    )

            if best_inner < self.best_score:
                self.best_score = best_inner
                self.best_code  = best_code_inner
                cur_code = best_code_inner
                self._save_model(best_inner)
                print(f'     ✅ Улучшение → MSE={self.best_score:.4f}')
            else:
                print(f'     → Без улучшения (best MSE={self.best_score:.4f})')

            self.tried_plans.append(plan)

        return cur_code

    def step_ensemble(self):
        print('\n' + '='*50)
        print('🎯 ШАГ 5: Ансамблирование')
        print('='*50)

        if len(self.models_reg) < 2:
            print('⚠️  Меньше 2 моделей — пропускаем')
            return

        m_info = '\n'.join(
            f'Model {i}: val_MSE={m["val_score"]:.4f}, '
            f'features={len(m["feature_cols"])}'
            for i, m in enumerate(self.models_reg)
        )
        print(f'  Моделей: {len(self.models_reg)}')

        for rnd in range(ENSEMBLE_ROUNDS):
            print(f'\n  Round {rnd+1}/{ENSEMBLE_ROUNDS}')
            plan = self.ensemble_agent.propose_plan(m_info)
            print(f'    Plan: {plan[:100]}')

            code = self.ensemble_agent.implement_plan(plan, m_info)
            self.ctx['models_registry'] = self.models_reg
            ok, out = self._run(code, f'Ensemble_r{rnd}')

            if ok:
                sc = extract_score(out)
                if sc is not None:
                    print(f'    MSE: {sc:.4f}')
                    MON.log(self.ensemble_agent.name, f'round_{rnd}', score=sc)
                    self.ensemble_agent.history.append({'plan': plan, 'score': sc})
                    if sc < self.best_score:
                        self.best_score = sc
                        print(f'    ✅ Новый лучший MSE: {sc:.4f}')

    def step_submission(self):
        print('\n' + '='*50)
        print('📤 ШАГ 6: Генерация submission.csv')
        print('='*50)

        df_test = self.ctx.get('df_test')
        if df_test is None:
            print('⚠️  df_test = None, пропускаем')
            return
        if 'best_model' not in self.ctx or 'feature_cols' not in self.ctx:
            print('⚠️  Модель не найдена в контексте')
            return

        try:
            model    = self.ctx['best_model']
            feat     = self.ctx['feature_cols']
            les      = self.ctx.get('label_encoders', {})

            avail = [c for c in feat if c in df_test.columns]
            X_test = df_test[avail].copy()

            for col, le in les.items():
                if col in X_test.columns:
                    X_test[col] = X_test[col].map(
                        lambda x, _le=le: _le.transform([str(x)])[0]
                        if str(x) in _le.classes_ else -1
                    )

            X_test = X_test.fillna(X_test.median(numeric_only=True))
            preds  = model.predict(X_test)
            id_col = df_test['_id'] if '_id' in df_test.columns else df_test.index

            sub = pd.DataFrame({'index': id_col, 'prediction': preds})
            sub.to_csv('submission.csv', index=False)
            print(f'✅ submission.csv сохранён ({len(sub)} строк)')
            print(f'   preds: min={preds.min():.3f} max={preds.max():.3f} '
                  f'mean={preds.mean():.3f}')
            MON.log('System', 'submission_saved')
        except Exception as e:
            print(f'❌ Ошибка submission: {e}')
            logger.error(traceback.format_exc())

    def run(self) -> float:
        print('\n' + '🚀 ' * 16)
        print('МУЛЬТИАГЕНТНАЯ СИСТЕМА — СТАРТ')
        print('🚀 ' * 16)

        self.step_clean()
        self.step_features()

        s1, code1 = self.step_train('LightGBM')
        s2, code2 = self.step_train('XGBoost')
        cur_code  = code1 if s1 <= s2 else code2

        if self.best_score < float('inf'):
            cur_code = self.step_refine(cur_code)
        else:
            print('⚠️  Пропуск рефайнмента — нет начального score')

        self.step_ensemble()
        self.step_submission()

        print(MON.report())
        return self.best_score
