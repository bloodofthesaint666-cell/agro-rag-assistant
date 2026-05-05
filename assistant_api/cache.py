"""
Модуль кеширования для RAG ассистента.
Использует SQLite для хранения пар вопрос-ответ с временными метками.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any
import os


class RAGCache:
    """Кеш для хранения результатов RAG запросов."""
    
    def __init__(self, db_path: str = "rag_cache.db"):
        """
        Инициализация кеша.
        
        Args:
            db_path: путь к файлу базы данных SQLite
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Создание таблицы кеша, если она не существует."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_query_hash(self, query: str) -> str:
        """
        Вычисление хеша запроса для использования как ключ кеша.
        
        Args:
            query: текст запроса
            
        Returns:
            SHA-256 хеш запроса
        """
        # Нормализация запроса: lowercase и удаление лишних пробелов
        normalized_query = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized_query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Получение ответа из кеша.
        
        Args:
            query: текст запроса
            
        Returns:
            Словарь с ответом и метаданными, или None если не найдено
        """
        query_hash = self._get_query_hash(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query, answer, context, created_at
            FROM cache
            WHERE query_hash = ?
        """, (query_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "query": result[0],
                "answer": result[1],
                "context": json.loads(result[2]) if result[2] else None,
                "created_at": result[3],
                "from_cache": True
            }
        
        return None
    
    def set(self, query: str, answer: str, context: list = None):
        """
        Сохранение ответа в кеш.
        
        Args:
            query: текст запроса
            answer: текст ответа
            context: список документов, использованных как контекст
        """
        query_hash = self._get_query_hash(query)
        context_json = json.dumps(context) if context else None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Используем INSERT OR REPLACE для обновления существующих записей
        cursor.execute("""
            INSERT OR REPLACE INTO cache (query_hash, query, answer, context, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (query_hash, query, answer, context_json, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def clear(self):
        """Очистка всего кеша."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cache")
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кеша.
        
        Returns:
            Словарь со статистикой
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM cache")
        dates = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates[0] else None,
            "newest_entry": dates[1] if dates[1] else None,
            "db_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }


if __name__ == "__main__":
    # Тестирование кеша
    cache = RAGCache("test_cache.db")
    
    # Сохранение
    cache.set(
        query="Что такое машинное обучение?",
        answer="Машинное обучение - это раздел искусственного интеллекта...",
        context=["doc1", "doc2"]
    )
    
    # Получение
    result = cache.get("Что такое машинное обучение?")
    print("Результат из кеша:", result)
    
    # Статистика
    stats = cache.get_stats()
    print("Статистика кеша:", stats)
    
    # Очистка тестовой БД
    import os
    if os.path.exists("test_cache.db"):
        os.remove("test_cache.db")

