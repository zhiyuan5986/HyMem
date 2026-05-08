
"""Embedding-based retrieval system for HyMem."""

import os
from typing import List, Dict, Optional, Any
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimpleEmbeddingRetriever:
    """Simple retrieval system using text embeddings for semantic search."""

    def __init__(self, model_name: str = '', api_key: str = '', base_url: str = ''):
        from llama_index.embeddings.openai import OpenAIEmbedding

        self.model = OpenAIEmbedding(model_name=model_name, api_base=base_url, api_key=api_key)
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

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str) -> None:
        os.makedirs(os.path.dirname(retriever_cache_file), exist_ok=True)
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        state = {'corpus': self.corpus, 'document_ids': self.document_ids}
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str) -> "SimpleEmbeddingRetriever":
        if os.path.exists(retriever_cache_embeddings_file):
            self.embeddings = np.load(retriever_cache_embeddings_file)
        if os.path.exists(retriever_cache_file):
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
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
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = None
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        texts = [d['content'] for d in documents]
        vectors = self.model.get_text_embedding_batch(texts)
        payloads = []
        start_idx = len(self.corpus)
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.corpus.append(doc['content'])
            self.document_ids[doc['content']] = idx
            payloads.append({'vector': vectors[i], 'content': doc['content'], 'index': idx, 'link': doc.get('link', ''), 'timestamp': doc.get('timestamp', '')})
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=payloads, mode='overwrite')
        else:
            self.table.add(payloads)

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

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str) -> None:
        return

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str) -> "LanceDBMemorySummaryRetriever":
        return self
