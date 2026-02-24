# RAG-система для технической документации ViPNet Coordinator HW 5

## Описание

RAG (Retrieval-Augmented Generation) система для поиска и ответов на вопросы по технической документации ViPNet Coordinator HW 5. Использует локальную LLM (Mistral-7B-Instruct, < 8B параметров) и векторную базу данных ChromaDB.

## Структура проекта

```
rag_project/
├── 01_data_extraction.py    # Извлечение текста из PDF и чанкинг
├── 02_build_index.py        # Построение векторного индекса (ChromaDB)
├── 03_rag_pipeline.py       # Основной RAG-пайплайн (поиск + генерация)
├── 04_generate_benchmark.py # Генерация синтетического бенчмарка
├── 05_evaluate.py           # Оценка качества поиска (Hit Rate, MRR)
├── rag_research.ipynb       # Jupyter-ноутбук с исследованиями
├── requirements.txt         # Зависимости проекта
├── data/
│   ├── chunks.json          # Извлечённые чанки из PDF
│   ├── benchmark.json       # Бенчмарк-датасет (вопросы + ответы)
│   ├── eval_results.csv     # Результаты оценки
│   └── chroma_db/           # Векторная база данных
└── models/
    └── mistral-7b-instruct-v0.3.Q4_K_M.gguf  # (скачать отдельно)
```

## Установка

### 1. Создание виртуального окружения

```powershell
python -m venv rag_env
.\rag_env\Scripts\Activate.ps1
```

### 2. Установка PyTorch (CUDA 12.4)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Установка остальных зависимостей

```powershell
pip install pymupdf sentence-transformers chromadb tqdm numpy jupyter ipykernel
```

### 4. Установка llama-cpp-python с поддержкой CUDA

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --no-cache-dir
```

### 5. Скачивание модели LLM

В случае отсутствия, скачайте `mistral-7b-instruct-v0.3.Q4_K_M.gguf` (~4.4 GB) с HuggingFace:
```
https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF
```
Поместите файл в папку `rag_project/models/`.

## Использование

### Шаг 1: Извлечение данных из PDF

```powershell
python 01_data_extraction.py
```
Создаёт `data/chunks.json` с ~N чанками из 6 PDF-документов.

### Шаг 2: Построение векторного индекса

```powershell
python 02_build_index.py
```
Загружает модель `multilingual-e5-large`, вычисляет эмбеддинги и сохраняет в ChromaDB.

### Шаг 3: Генерация бенчмарка

```powershell
python 04_generate_benchmark.py
```
Создаёт `data/benchmark.json` с 20 вопросами по документации.

### Шаг 4: Оценка качества поиска

```powershell
python 05_evaluate.py
```
Выводит метрики Hit Rate @ K и MRR, сохраняет `data/eval_results.csv`.

### Шаг 5: Интерактивный RAG-чат

```powershell
python 03_rag_pipeline.py
```
Запускает интерактивный режим вопрос-ответ.

### Jupyter-ноутбук с исследованиями

```powershell
python -m ipykernel install --user --name=rag_env --display-name "Python 3 (rag_env)"
jupyter notebook rag_research.ipynb
```

## Архитектура системы

```
Вопрос пользователя
       │
       ▼
[Embedding Model]  ←  intfloat/multilingual-e5-large
       │
       ▼
[Vector Search]    ←  ChromaDB (cosine similarity)
       │
       ▼
[Top-K Chunks]     ←  K=5 релевантных фрагментов
       │
       ▼
[LLM Generation]   ←  Mistral-7B-Instruct-v0.3 (GGUF, Q4_K_M)
       │
       ▼
    Ответ
```