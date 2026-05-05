"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
from glob import glob
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


# Базовая метадата по типу документа.
# Используется для семантической фильтрации (topic / process / intent / role).
# Значения должны быть примитивами (требование ChromaDB), поэтому роли — строкой.
DOC_METADATA: Dict[str, Dict[str, Any]] = {
    "reglament_service_notes_sap.txt": {
        "topic": "служебные записки",
        "process": "SAP",
        "intent": "согласование изменений",
        "role": "инициатор, агроном, руководитель, экономист",
    },
    "reglament_meetings_do.txt": {
        "topic": "планерки",
        "process": "управление ДО",
        "intent": "оперативное планирование",
        "role": "руководитель ДО, агроном, инженер, экономист, диспетчер",
    },
    "reglament_agro_season_sap.txt": {
        "topic": "агросезон",
        "process": "SAP",
        "intent": "управление агросезоном",
        "role": "экономист, агроном",
    },
}


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # OpenAI клиент для создания embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents(self, file_path: str):
        """
        Загрузка документов из файла или директории в векторное хранилище.
        
        Args:
            file_path: путь к файлу с документами или директории с `.txt` файлами
        """
        # Проверка существования файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Путь {file_path} не найден")

        # Собираем пары (исходный файл, текст), чтобы сохранить связь чанк -> источник
        sources: List[tuple] = []
        if os.path.isdir(file_path):
            text_files = sorted(glob(os.path.join(file_path, "*.txt")))
            if not text_files:
                raise FileNotFoundError(f"В директории {file_path} не найдено .txt файлов")

            print(f"Найдено {len(text_files)} текстовых файлов в {file_path}")
            for text_file in text_files:
                with open(text_file, 'r', encoding='utf-8') as f:
                    file_text = f.read().strip()
                    if file_text:
                        sources.append((text_file, file_text))
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_text = f.read()
            sources.append((file_path, file_text))

        # Проверка, не загружены ли уже документы
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        documents: List[str] = []
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        chunk_idx = 0
        for src_path, src_text in sources:
            file_chunks = self._chunk_text(src_text)
            source_name = os.path.basename(src_path)
            file_meta: Dict[str, Any] = {"source": source_name}
            file_meta.update(DOC_METADATA.get(source_name, {}))
            print(f"  {source_name}: {len(file_chunks)} чанков (topic={file_meta.get('topic', '—')})")

            for chunk in file_chunks:
                embedding = self._create_embedding(chunk)

                documents.append(chunk)
                ids.append(f"doc_{chunk_idx}")
                embeddings.append(embedding)
                metadatas.append(dict(file_meta))

                chunk_idx += 1
                if chunk_idx % 10 == 0:
                    print(f"Обработано {chunk_idx} чанков")

        if not documents:
            print("Не найдено непустых чанков для загрузки")
            return

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"Загружено {len(documents)} документов в коллекцию '{self.collection_name}'")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        where: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.

        Args:
            query: текст запроса
            top_k: количество документов для возврата
            where: опциональный metadata-фильтр для ChromaDB
                   (например, {"topic": "служебные записки"})

        Returns:
            список документов с метаданными
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        documents = []
        if results['documents'] and len(results['documents']) > 0:
            metadatas_list = results.get('metadatas') or [[]]
            for i in range(len(results['documents'][0])):
                metadata = {}
                if metadatas_list and metadatas_list[0] and i < len(metadatas_list[0]):
                    metadata = metadatas_list[0][i] or {}
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'metadata': metadata,
                })

        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    if os.path.isdir("data") or os.path.exists("data"):
        vector_store.load_documents("data")
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

