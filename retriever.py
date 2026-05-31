"""
SporraMom W3 智能客服系统 — 混合检索引擎

设计思路（面试要点）:
  1. BM25 关键词检索
     - 优势: 精确匹配产品型号、参数等关键词 (如 "1400mAh", "IP22")
     - 实现: jieba 中文分词 + BM25Okapi
  2. 向量语义检索
     - 优势: 理解用户意图 (如 "充不了电" → 匹配 "无法充电" 故障排除)
     - 实现: bge-small-zh 编码 + FAISS
  3. RRF (Reciprocal Rank Fusion) 融合
     - 为什么用 RRF: 不依赖不同检索器的分数绝对值，只看排名
     - 公式: score(doc) = Σ 1/(k + rank_i)，k=60 是论文推荐值
  4. Reranker 精排
     - 对 RRF 融合后的 top_k 候选做交叉编码打分
     - 比 embedding 的双塔模型更精准 (但更慢，所以只对 top_k 做)
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi
import faiss

import config
from data_processor import Chunk


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk: Chunk
    bm25_rank: int        # BM25 排名 (0-based)
    vector_rank: int      # 向量检索排名 (0-based)
    rrf_score: float      # RRF 融合分数
    rerank_score: float   # Reranker 分数 (仅最终结果集有)


class BM25Index:
    """BM25 关键词索引"""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        # jieba 分词
        self.tokenized_corpus = [
            list(jieba.cut(chunk.content)) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        返回 [(chunk_index, score), ...] 按分数降序
        """
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        # 取 top_k 的索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class VectorIndex:
    """向量语义索引"""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.model = None
        self.index = None
        self.embeddings = None

    def build(self, model_name: str = None):
        """构建向量索引"""
        from sentence_transformers import SentenceTransformer

        model_name = model_name or config.EMBEDDING_MODEL
        print(f"[VectorIndex] 加载 Embedding 模型: {model_name}")
        self.model = SentenceTransformer(model_name)

        texts = [chunk.content for chunk in self.chunks]
        print(f"[VectorIndex] 编码 {len(texts)} 个 chunks...")
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化 → 内积 = 余弦相似度
            show_progress_bar=True,
            batch_size=32,
        )

        # 构建 FAISS 索引 (内积)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
        print(f"[VectorIndex] FAISS 索引构建完成, dim={dim}, n={self.index.ntotal}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        返回 [(chunk_index, score), ...] 按相似度降序
        """
        if self.model is None or self.index is None:
            raise RuntimeError("向量索引未构建，请先调用 build()")

        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS 返回 -1 表示无效
                results.append((int(idx), float(score)))
        return results

    def save(self, path: str):
        """保存索引到磁盘"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        np.save(str(save_path / "embeddings.npy"), self.embeddings)

    def load(self, path: str):
        """从磁盘加载索引"""
        save_path = Path(path)
        self.index = faiss.read_index(str(save_path / "faiss.index"))
        self.embeddings = np.load(str(save_path / "embeddings.npy"))


class HybridRetriever:
    """混合检索器: BM25 + 向量 + RRF + Rerank"""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.bm25_index: Optional[BM25Index] = None
        self.vector_index: Optional[VectorIndex] = None
        self.reranker = None

    def build_indexes(self):
        """构建所有索引"""
        print("[HybridRetriever] 构建 BM25 索引...")
        self.bm25_index = BM25Index(self.chunks)

        print("[HybridRetriever] 构建向量索引...")
        self.vector_index = VectorIndex(self.chunks)
        self.vector_index.build()

    def _load_reranker(self):
        """延迟加载 Reranker (首次使用时才加载)"""
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"[HybridRetriever] 加载 Reranker: {config.RERANKER_MODEL}")
            self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        k: int = None,
    ) -> List[Tuple[int, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合两个排序。

        公式: score(doc) = Σ 1/(k + rank_i)
        - rank_i 是文档在第 i 个检索器中的排名 (1-based)
        - k 是平滑常数 (默认 60，论文推荐值)

        优势:
        - 不依赖分数绝对值 (BM25 分数和向量分数量纲不同)
        - 只看排名，鲁棒性强
        """
        k = k or config.RRF_K
        score_map: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(bm25_results, start=1):
            score_map[idx] = score_map.get(idx, 0) + 1.0 / (k + rank)

        for rank, (idx, _) in enumerate(vector_results, start=1):
            score_map[idx] = score_map.get(idx, 0) + 1.0 / (k + rank)

        # 按 RRF 分数降序排序
        sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    def _rerank(self, query: str, candidates: List[Tuple[int, float]], top_n: int) -> List[Tuple[int, float]]:
        """
        用 CrossEncoder 对候选文档重排序。

        为什么需要 Rerank:
        - Embedding 双塔模型: query 和 doc 独立编码，交互信息有限
        - CrossEncoder: query 和 doc 拼接输入，交叉注意力，更精准
        - 代价: 更慢，所以只对 top_k 候选做
        """
        self._load_reranker()

        pairs = [(query, self.chunks[idx].content) for idx, _ in candidates]
        scores = self.reranker.predict(pairs)

        # 按 rerank 分数降序
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_n]

    def retrieve(
        self,
        query: str,
        bm25_top_k: int = None,
        vector_top_k: int = None,
        rerank_top_n: int = None,
    ) -> List[RetrievalResult]:
        """
        完整检索流程:
        1. BM25 关键词检索 → top_k 候选
        2. 向量语义检索 → top_k 候选
        3. RRF 融合 → 合并排序
        4. Reranker 精排 → 最终 top_n 结果

        返回: RetrievalResult 列表，含各阶段排名和分数
        """
        bm25_top_k = bm25_top_k or config.BM25_TOP_K
        vector_top_k = vector_top_k or config.VECTOR_TOP_K
        rerank_top_n = rerank_top_n or config.RERANK_TOP_N

        # Step 1: BM25
        bm25_results = self.bm25_index.search(query, bm25_top_k)
        bm25_rank_map = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}

        # Step 2: 向量
        vector_results = self.vector_index.search(query, vector_top_k)
        vector_rank_map = {idx: rank for rank, (idx, _) in enumerate(vector_results)}

        # Step 3: RRF 融合
        rrf_results = self._rrf_fusion(bm25_results, vector_results)

        # Step 4: Rerank
        rerank_candidates = rrf_results[:bm25_top_k + vector_top_k]  # 取融合后的前 N 候选
        reranked = self._rerank(query, rerank_candidates, rerank_top_n)

        # 组装结果
        results = []
        for idx, rerank_score in reranked:
            chunk = self.chunks[idx]
            rrf_score = next((s for i, s in rrf_results if i == idx), 0.0)
            results.append(RetrievalResult(
                chunk=chunk,
                bm25_rank=bm25_rank_map.get(idx, -1),
                vector_rank=vector_rank_map.get(idx, -1),
                rrf_score=rrf_score,
                rerank_score=rerank_score,
            ))

        return results


def build_retriever(chunks: List[Chunk]) -> HybridRetriever:
    """构建并返回检索器实例"""
    retriever = HybridRetriever(chunks)
    retriever.build_indexes()
    return retriever


if __name__ == "__main__":
    from data_processor import load_or_build_chunks

    chunks = load_or_build_chunks()
    retriever = build_retriever(chunks)

    # 测试检索
    test_queries = [
        "吸奶器充不了电怎么办",
        "如何选择合适的喇叭罩尺寸",
        "母乳可以保存多久",
        "IP22 是什么意思",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}")
        results = retriever.retrieve(query)
        for i, r in enumerate(results):
            print(f"  [{i+1}] Rerank={r.rerank_score:.3f} | BM25#{r.bm25_rank+1} Vec#{r.vector_rank+1} | {r.chunk.section_path}")
            print(f"      {r.chunk.content[:100]}...")
