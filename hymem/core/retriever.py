
"""Embedding-based retrieval system for HyMem."""

import os
from typing import List, Dict, Optional, Any
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from hymem.core.memory import MemorySummary


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
                [MemorySummary(content=doc).to_dict() for doc in self.corpus],
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
        self.table = self.db.open_table(table_name) if table_name in self.db.table_names() else self.db.create_table(table_name, data=[])
        self.entries: List[MemorySummary] = self.get_all_entries()

    def get_all_entries(self) -> List[MemorySummary]:
        if self.table is None:
            return []
        rows = self.table.to_list()
        return [
            MemorySummary(
                content=row.get('content', ''),
                link=row.get('link', ''),
                timestamp=row.get('timestamp', ''),
                metadata=row.get('metadata', {}),
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
                'index': start_idx + i,
                'link': doc.get('link', ''),
                'timestamp': doc.get('timestamp', ''),
                'metadata': doc.get('metadata', {}),
            }
            payloads.append(entry)
        self.table.add(payloads)
        self.entries.extend([
            MemorySummary(
                content=doc.get('content', ''),
                link=doc.get('link', ''),
                timestamp=doc.get('timestamp', ''),
                metadata=doc.get('metadata', {}),
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
        rs = self.table.search(query, query_type='fts').limit(k).to_list()
        return np.array([int(r['index']) for r in rs])

    def search(self, query: str, k: int = 5) -> np.ndarray:
        return self.semantic_search(query, k)

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: Optional[str] = None) -> None:
        return

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: Optional[str] = None) -> "LanceDBMemorySummaryRetriever":
        self.entries = self.get_all_entries()
        return self
