"""
Консольное приложение для взаимодействия с RAG ассистентом (API mode).
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Загрузка переменных окружения из .env файла
# Ищем .env в корне проекта (на уровень выше)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║         RAG Ассистент (API Mode)                        ║
║  Retrieval-Augmented Generation через OpenAI API        ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")
    
    # Индикатор источника ответа
    if result['from_cache']:
        print("💾 Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"🌐 Источник: OpenAI API ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\n💬 Ответ:\n{result['answer']}")

    sources = _extract_sources(result.get('context_docs'))
    if sources:
        print("\n📚 Источники:")
        for i, src in enumerate(sources, 1):
            print(f"   {i}. {src}")

    print(f"{'─'*60}\n")


def _extract_sources(context_docs) -> list:
    """
    Достаёт человекочитаемые названия источников из чанков.

    Поддерживает три формата элементов:
    - dict с metadata (новый формат от vector_store)
    - dict без metadata (старые записи)
    - строка (легаси-кеш с одним лишь текстом)
    """
    if not context_docs:
        return []

    sources = []
    for doc in context_docs:
        if isinstance(doc, str):
            label = "из кеша"
        else:
            meta = (doc.get('metadata') or {}) if isinstance(doc, dict) else {}
            label = (
                meta.get('source')
                or meta.get('title')
                or meta.get('section')
                or (doc.get('source') if isinstance(doc, dict) else None)
                or (doc.get('id') if isinstance(doc, dict) else None)
                or "документ"
            )
        sources.append(label)
    return sources


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()
    
    print(f"\n{'═'*60}")
    print("📊 СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═'*60}")
    
    print("\n🗄️  Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\n💾 Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    
    print(f"\n🤖 Модель: {stats['model']}")
    print(f"🌐 Режим: {stats['mode']}")
    print(f"{'═'*60}\n")


def main():
    """Главная функция приложения."""
    print_banner()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: переменная окружения OPENAI_API_KEY не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    try:
        # Инициализация RAG pipeline
        print("🚀 Инициализация системы...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data",
            model="gpt-4o-mini"
        )
        print("\n✅ Система готова к работе!\n")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)
    
    # Основной цикл взаимодействия
    while True:
        try:
            # Получение запроса от пользователя
            user_input = input("💭 Ваш вопрос: ").strip()
            
            # Обработка специальных команд
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if user_input.lower() == 'stats':
                print_stats(pipeline)
                continue
            
            if user_input.lower() == 'clear':
                confirm = input("⚠️  Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    print("✅ Кеш очищен")
                continue
            
            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue
            
            # Обработка запроса через RAG pipeline
            result = pipeline.query(user_input)
            
            # Вывод результата
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()

