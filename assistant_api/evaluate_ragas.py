"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate

# Правильный импорт для RAGAS 0.4.x - используем классы метрик
try:
    # Новый способ импорта (RAGAS 0.4+)
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision
    faithfulness = Faithfulness
    context_precision = ContextPrecision
except ImportError:
    try:
        # Альтернативный импорт из collections
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        # Fallback на старый импорт
        from ragas.metrics import faithfulness, context_precision

from rag_pipeline import RAGPipeline


# Тестовые вопросы для оценки RAG системы
EVALUATION_QUESTIONS = [
    "Что такое машинное обучение?",
    "Какие основные типы машинного обучения существуют?",
    "Что такое нейронная сеть?",
    "Как работают трансформеры в NLP?",
    "Что такое RAG и как он работает?"
]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data",
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)
    
    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision")
    print("   (Answer Relevancy отключена из-за несовместимости embeddings API)")
    print("   (это займёт 1-2 минуты, так как RAGAS использует OpenAI для оценки)\n")
    
    # Используем только метрики, которые работают без проблем с embeddings
    # ВАЖНО: инициализируем метрики как объекты (с круглыми скобками)
    print("   [+] Используем базовые метрики RAGAS\n")
    metrics_to_use = [faithfulness(), context_precision()]
    
    # Запускаем оценку RAGAS
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)
    
    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    # Вычисляем средние значения метрик (игнорируя NaN)
    import math
    
    faithfulness_values = [
        v for v in result['faithfulness'] 
        if not (isinstance(v, float) and math.isnan(v))
    ]
    context_precision_values = [
        v for v in result['context_precision'] 
        if not (isinstance(v, float) and math.isnan(v))
    ]
    
    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values) 
        if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values) 
        if context_precision_values else 0
    )
    
    # Выводим общие метрики
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    
    # Вычисляем и выводим средний балл
    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    # Оценка качества системы
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)
    
    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i+1}. {question}")
        
        # Faithfulness
        faith_val = result['faithfulness'][i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print(f"   Faithfulness:       не удалось вычислить")
        
        # Context Precision
        cp_val = result['context_precision'][i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print(f"   Context Precision:  не удалось вычислить")
    
    # Пояснения к метрикам
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

ПРИМЕЧАНИЕ:
  Answer Relevancy временно отключена из-за несовместимости embeddings API
  в текущей версии RAGAS с langchain-openai.
    """)
    
    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()

