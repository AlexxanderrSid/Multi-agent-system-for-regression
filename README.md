# 🤖 Multi-Agent Regression System

Мультиагентная система для решения задач регрессии на табличных данных.  
Архитектура вдохновлена **MLE-STAR** (Search & Targeted Refinement).  
LLM-бэкенд: **Qwen2.5-Coder-7B-Instruct** (open-source, Apache 2.0), запускаемый локально через `llama-cpp-python`.

---

## Архитектура системы

```
DataAnalystAgent  →  FeatureEngineerAgent  →  ModelAgent (LightGBM + XGBoost)
                                                        │
                              ┌─────────────────────────┘
                              ▼
                    OUTER LOOP (×3 шага):
                      AblationAgent → summary → plan
                        INNER LOOP (×5 шагов):
                          ModelAgent.run_refined(plan) → evaluate (MSE)
                          FeatureEngineerAgent.propose_new_plan
                              │
                    EnsembleAgent (×5 раундов)
                              │
                         submission.csv
```

| Агент | Роль |
|---|---|
| `DataAnalystAgent` | EDA, очистка данных |
| `FeatureEngineerAgent` | Feature engineering + RAG + plan trajectory |
| `ModelAgent` | Обучение LightGBM/XGBoost, рефайнмент |
| `AblationAgent` | Ablation study → целевой компонент (MLE-STAR) |
| `EnsembleAgent` | Итеративное ансамблирование |
| `DebuggerAgent` | Автоматическое исправление ошибок |

---

## Структура репозитория

```
multi-agent-regression/
│
├── main.py                        # Точка входа: загрузка модели, данных, запуск
├── requirements.txt
│
├── data/                          # Данные (не коммитить в git)
│   ├── train.csv
│   └── test.csv
│
├── models/                        # GGUF-модель (скачивается автоматически)
│
├── docs/
│   └── methodology.md             # Описание и обоснование методов
│
└── src/
    ├── config.py                  # Гиперпараметры системы, логирование
    ├── system.py                  # MultiAgentMLSystem — главный оркестратор
    ├── benchmark.py               # Бенчмарк базовых моделей (5-fold CV)
    │
    ├── core/
    │   ├── guardrails.py          # Валидация входных данных, safety-проверки
    │   ├── executor.py            # Безопасное выполнение кода, extract_score
    │   ├── monitor.py             # Monitor — логирование событий и метрик
    │   └── llm_client.py          # LLMClient — обёртка над llama_cpp.Llama
    │
    ├── rag/
    │   └── knowledge_base.py      # KnowledgeBase — keyword-based RAG
    │
    └── agents/
        ├── data_analyst.py        # DataAnalystAgent
        ├── feature_engineer.py    # FeatureEngineerAgent
        ├── model_agent.py         # ModelAgent
        ├── ablation_agent.py      # AblationAgent
        ├── ensemble_agent.py      # EnsembleAgent
        └── debugger_agent.py      # DebuggerAgent
```

---

## Требования

- Python 3.10+
- CUDA-совместимый GPU (рекомендуется ≥16 GB VRAM, например NVIDIA T4/A100)
- Colab, Kaggle Notebook или локальная машина с GPU

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/<your-username>/multi-agent-regression.git
cd multi-agent-regression
```

### 2. Установить зависимости

**С поддержкой CUDA (рекомендуется для Colab/Kaggle T4):**

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.4 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
pip install -r requirements.txt
```

**Без GPU (CPU-only, значительно медленнее):**

```bash
pip install llama-cpp-python==0.3.4
pip install -r requirements.txt
```

### 3. Подготовить данные

Поместите файлы в папку `data/`:

```
data/
├── train.csv   # должен содержать колонку 'target' (числовой тип)
└── test.csv
```

При работе в Kaggle раскомментируйте соответствующие строки в `main.py`:

```python
# df_train = pd.read_csv('/kaggle/input/mws-ai-agents-2026/train.csv')
# df_test  = pd.read_csv('/kaggle/input/mws-ai-agents-2026/test.csv')
```

---

## Запуск

```bash
python main.py
```

При первом запуске автоматически скачается модель (~4.7 GB):

```
models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

### Ожидаемый вывод

```
🚀 🚀 🚀 МУЛЬТИАГЕНТНАЯ СИСТЕМА — СТАРТ
==================================================
📊 ШАГ 1: Очистка данных
...
🔧 ШАГ 2: Feature Engineering
...
🤖 ШАГ 3: Обучение (LightGBM)
  MSE = 1234.5678
...
🔬 ШАГ 4: Ablation + Refinement
...
🎯 ШАГ 5: Ансамблирование
...
📤 ШАГ 6: Генерация submission.csv
✅ submission.csv сохранён

=======================================================
📊 ИТОГОВЫЙ ОТЧЁТ
=======================================================
🏆 Лучший MSE: ...
```

Результат сохраняется в `submission.csv`.

---

## Настройка гиперпараметров

Все ключевые параметры системы находятся в `src/config.py`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `OUTER_LOOP_STEPS` | 3 | Количество outer-итераций рефайнмента |
| `INNER_LOOP_STEPS` | 5 | Количество inner-планов на каждый outer-шаг |
| `MAX_DEBUG` | 3 | Максимум попыток авто-дебага |
| `ENSEMBLE_ROUNDS` | 5 | Раундов ансамблирования |
| `RANDOM_STATE` | 42 | Seed для воспроизводимости |

---

## Безопасность

Система содержит несколько уровней защиты от некорректного кода, генерируемого LLM:

- **`check_code_safety`** — блокирует `os.system`, `subprocess`, запись файлов
- **`check_data_leakage`** — предупреждает о `fit` на тестовых данных
- **`validate_input_data`** — проверяет корректность входного DataFrame
- **`DebuggerAgent`** — автоматически исправляет ошибки выполнения (до `MAX_DEBUG` попыток)

---

## Метрика

Все агенты, мониторинг и отбор лучшей модели ориентированы на **MSE (Mean Squared Error)** — чем меньше, тем лучше.

---

## Лицензия

Код системы — MIT.  
Модель Qwen2.5-Coder-7B — [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).
