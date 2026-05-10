
"""Embedding-based retrieval system for HyMem."""

import os
import json
import time
from typing import List, Dict, Optional, Any
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from hymem.core.memory import MemorySummary, LLMSpan


class SimpleEmbeddingRetriever:
    """Simple retrieval system using text embeddings for semantic search."""

    def __init__(self, model_name: str = '', api_key: str = '', base_url: str = '', embed_backend: str = "openai"):
        self.embed_backend = embed_backend
        if embed_backend == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            self.model = OpenAIEmbedding(model_name=model_name, api_base=base_url, api_key=api_key)
        elif embed_backend == "huggingface":
            import torch
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self.model = HuggingFaceEmbedding(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError(f"Unsupported embed_backend: {embed_backend}")
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}

    def add_documents(self, documents: List[str]) -> None:
        if not self.corpus:
            self.corpus = documents
            self.embeddings = np.array(self.model.get_text_embedding_batch(documents))
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = np.array(self.model.get_text_embedding_batch(documents))
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> np.ndarray:
        if not self.corpus or self.embeddings is None:
            return np.array([])
        query_embedding = self.model.get_text_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return np.argsort(similarities)[-k:][::-1]

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(retriever_cache_file), exist_ok=True)
        if self.embeddings is not None and retriever_cache_embeddings_file is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        state = {'corpus': self.corpus, 'document_ids': self.document_ids}
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
        summaries_file = os.path.join(os.path.dirname(retriever_cache_file), "summaries.pkl")
        with open(summaries_file, 'wb') as f:
            pickle.dump(
                [MemorySummary(content=doc).model_dump() for doc in self.corpus],
                f
            )

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: Optional[str] = None) -> "SimpleEmbeddingRetriever":
        if retriever_cache_embeddings_file is not None and os.path.exists(retriever_cache_embeddings_file):
            self.embeddings = np.load(retriever_cache_embeddings_file)
        if os.path.exists(retriever_cache_file):
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state.get('corpus', [])
                self.document_ids = state.get('document_ids', {})
        summaries_file = os.path.join(os.path.dirname(retriever_cache_file), "summaries.pkl")
        if os.path.exists(summaries_file):
            with open(summaries_file, 'rb') as f:
                summaries = pickle.load(f)
            self.corpus = [s.get("content", "") for s in summaries]
        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, api_key: str = '', base_url: str = '') -> "SimpleEmbeddingRetriever":
        all_docs = []
        for m in memories.values():
            metadata_text = f"{getattr(m, 'context', '')} {' '.join(getattr(m, 'keywords', []))} {' '.join(getattr(m, 'tags', []))}"
            all_docs.append(f"{m.content} , {metadata_text}")
        retriever = cls(model_name, api_key, base_url)
        retriever.add_documents(all_docs)
        return retriever


class LanceDBMemorySummaryRetriever:
    """Retriever backed by local embedding model + LanceDB for MemorySummary search."""

    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', db_path: str = './lancedb', table_name: str = 'memory_summary'):
        import lancedb
        import torch
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.model = HuggingFaceEmbedding(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
        else:
            self.table = self.db.create_table(table_name, data=[], schema=self._get_table_schema())
        self.entries: List[MemorySummary] = self.get_all_entries()
        self._ensure_fts_index()

    def _get_table_schema(self):
        """Build LanceDB schema so empty-table initialization is valid."""
        import pyarrow as pa

        vector_dim = self._get_vector_dim()
        return pa.schema([
            pa.field('vector', pa.list_(pa.float32(), vector_dim)),
            pa.field('content', pa.string()),
            pa.field('id', pa.string()),
            pa.field('index', pa.int64()),
            pa.field('link', pa.string()),
            pa.field('timestamp', pa.string()),
            pa.field('metadata', pa.string()),
        ])

    @staticmethod
    def _serialize_metadata(metadata: Any) -> str:
        if isinstance(metadata, str):
            return metadata
        return json.dumps(metadata or {}, ensure_ascii=False)

    @staticmethod
    def _parse_metadata(metadata: Any) -> Dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _get_vector_dim(self) -> int:
        """Infer embedding dimensionality from model internals or a probe call."""
        sentence_model = getattr(self.model, '_model', None)
        if sentence_model is not None:
            get_dim = getattr(sentence_model, 'get_sentence_embedding_dimension', None)
            if callable(get_dim):
                dim = get_dim()
                if isinstance(dim, int) and dim > 0:
                    return dim

        probe = self.model.get_text_embedding('schema_probe')
        if not probe:
            raise ValueError('Unable to infer embedding dimension for LanceDB schema.')
        return len(probe)


    def get_all_entries(self) -> List[MemorySummary]:
        if self.table is None:
            return []
        rows = self.table.to_arrow().to_pylist()
        return [
            MemorySummary(
                content=row.get('content', ''),
                id=row.get('id', ''),
                link=row.get('link', ''),
                timestamp=row.get('timestamp', ''),
                metadata=self._parse_metadata(row.get('metadata', '{}')),
            )
            for row in rows
        ]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        texts = [d['content'] for d in documents]
        vectors = self.model.get_text_embedding_batch(texts)
        start_idx = len(self.entries)
        payloads = []
        for i, doc in enumerate(documents):
            entry = {
                'vector': vectors[i],
                'content': doc['content'],
                'id': doc.get('id', ''),
                'index': start_idx + i,
                'link': doc.get('link', ''),
                'timestamp': doc.get('timestamp', ''),
                'metadata': self._serialize_metadata(doc.get('metadata', {})),
            }
            payloads.append(entry)
        self.table.add(payloads)
        self._ensure_fts_index()
        self.entries.extend([
            MemorySummary(
                content=doc.get('content', ''),
                id=doc.get('id', ''),
                link=doc.get('link', ''),
                timestamp=doc.get('timestamp', ''),
                metadata=self._parse_metadata(self._serialize_metadata(doc.get('metadata', {}))),
            )
            for doc in documents
        ])

    def semantic_search(self, query: str, k: int = 5) -> np.ndarray:
        if self.table is None:
            return np.array([])
        qv = self.model.get_text_embedding(query)
        rs = self.table.search(qv).limit(k).to_list()
        return np.array([int(r['index']) for r in rs])

    def keyword_search(self, query: str, k: int = 5) -> np.ndarray:
        if self.table is None:
            return np.array([])
        self._ensure_fts_index()
        rs = self.table.search(query, query_type='fts').limit(k).to_list()
        return np.array([int(r['index']) for r in rs])

    def _ensure_fts_index(self, retries: int = 20, wait_seconds: float = 0.1) -> None:
        """Create FTS index safely across concurrent workers/processes."""
        if self.table is None:
            return
        lock_path = os.path.join(os.path.dirname(self.db.uri), f".{self.table_name}_fts.lock")
        for attempt in range(retries):
            lock_fd = None
            acquired = False
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(lock_fd)
                lock_fd = None
                acquired = True
                self.table.create_fts_index("content", replace=True)
                return
            except FileExistsError:
                time.sleep(wait_seconds)
            except Exception:
                return
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
                if acquired and os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass

    def search(self, query: str, k: int = 5) -> np.ndarray:
        return self.semantic_search(query, k)

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: Optional[str] = None) -> None:
        return

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: Optional[str] = None) -> "LanceDBMemorySummaryRetriever":
        self.entries = self.get_all_entries()
        return self

    def get_entry_by_id(self, entry_id: str) -> Optional[MemorySummary]:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None


class LanceDBLLMSpanRetriever:
    """Retriever/storage for fine-grained LLM spans."""

    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', db_path: str = './lancedb', table_name: str = 'llm_spans'):
        import lancedb
        import torch
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.model = HuggingFaceEmbedding(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
        else:
            self.table = self.db.create_table(table_name, data=[], schema=self._get_table_schema())
        self.entries: List[LLMSpan] = self.get_all_entries()
        self._ensure_fts_index()

    def _get_table_schema(self):
        import pyarrow as pa
        vector_dim = len(self.model.get_text_embedding('schema_probe'))
        return pa.schema([
            pa.field('vector', pa.list_(pa.float32(), vector_dim)),
            pa.field('content', pa.string()),
            pa.field('id', pa.string()),
            pa.field('index', pa.int64()),
            pa.field('timestamp', pa.string()),
            pa.field('metadata', pa.string()),
        ])

    @staticmethod
    def _serialize_metadata(metadata: Any) -> str:
        if isinstance(metadata, str):
            return metadata
        return json.dumps(metadata or {}, ensure_ascii=False)

    @staticmethod
    def _parse_metadata(metadata: Any) -> Dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def get_all_entries(self) -> List[LLMSpan]:
        rows = self.table.to_arrow().to_pylist()
        return [LLMSpan(content=r.get('content', ''), id=r.get('id', ''), timestamp=r.get('timestamp', ''), metadata=self._parse_metadata(r.get('metadata', '{}'))) for r in rows]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        vectors = self.model.get_text_embedding_batch([d['content'] for d in documents])
        start_idx = len(self.entries)
        payloads = []
        for i, doc in enumerate(documents):
            payloads.append({
                'vector': vectors[i], 'content': doc['content'], 'id': doc.get('id', ''), 'index': start_idx + i,
                'timestamp': doc.get('timestamp', ''), 'metadata': self._serialize_metadata(doc.get('metadata', {})),
            })
        self.table.add(payloads)
        self._ensure_fts_index()
        self.entries.extend([LLMSpan(content=d.get('content', ''), id=d.get('id', ''), timestamp=d.get('timestamp', ''), metadata=self._parse_metadata(self._serialize_metadata(d.get('metadata', {})))) for d in documents])

    def search(self, query: str, k: int = 5) -> np.ndarray:
        qv = self.model.get_text_embedding(query)
        rs = self.table.search(qv).limit(k).to_list()
        return np.array([int(r['index']) for r in rs])

    def keyword_search(self, query: str, k: int = 5) -> np.ndarray:
        self._ensure_fts_index()
        rs = self.table.search(query, query_type='fts').limit(k).to_list()
        return np.array([int(r['index']) for r in rs])

    def _ensure_fts_index(self, retries: int = 20, wait_seconds: float = 0.1) -> None:
        if self.table is None:
            return
        lock_path = os.path.join(os.path.dirname(self.db.uri), f".{self.table_name}_fts.lock")
        for attempt in range(retries):
            lock_fd = None
            acquired = False
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(lock_fd)
                lock_fd = None
                acquired = True
                self.table.create_fts_index("content", replace=True)
                return
            except FileExistsError:
                time.sleep(wait_seconds)
            except Exception:
                return
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
                if acquired and os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass

    def get_entry_by_id(self, entry_id: str) -> Optional[LLMSpan]:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None
