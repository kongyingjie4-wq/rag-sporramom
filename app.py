"""
SporraMom W3 智能客服系统 — 全链路调试界面

设计目标: 输入一个问题，能看到 RAG 流程中每一步的中间数据。
"""

import time
import json
from typing import List, Tuple, Dict, Any

import gradio as gr
import numpy as np

import config
from data_processor import load_or_build_chunks, get_chunks_summary, Chunk
from retriever import HybridRetriever, build_retriever
from rag_pipeline import (
    run_rag, RAGResponse, _judge_intent, _format_context,
    _generate_answer, _self_check, SYSTEM_PROMPT, ANSWER_PROMPT,
    INTENT_PROMPT
)
from llm_client import get_llm_client

# ============================================================
# 全局状态
# ============================================================
_retriever: HybridRetriever = None
_chunks: List[Chunk] = None


def init_system():
    """初始化系统"""
    global _retriever, _chunks

    print("=" * 60)
    print("SporraMom W3 智能客服系统 — 初始化中...")
    print("=" * 60)

    _chunks = load_or_build_chunks()
    summary = get_chunks_summary(_chunks)
    print(f"数据摘要: {summary['total_chunks']} 个 chunks, {summary['total_chars']} 字符")

    _retriever = build_retriever(_chunks)

    print("=" * 60)
    print("初始化完成！")
    print("=" * 60)

    return summary


# ============================================================
# 全链路调试: 每一步都返回详细数据
# ============================================================

def debug_full_pipeline(query: str):
    """
    执行完整 RAG 流程，返回每一步的详细中间数据。
    """
    if not query.strip():
        return {k: "请输入问题" for k in [
            "step1_intent", "step2_bm25", "step3_vector",
            "step4_rrf", "step5_rerank", "step6_relevance",
            "step7_prompt", "step8_answer", "step9_selfcheck",
            "final_answer"
        ]}

    if _retriever is None:
        return {k: "系统未初始化" for k in [
            "step1_intent", "step2_bm25", "step3_vector",
            "step4_rrf", "step5_rerank", "step6_relevance",
            "step7_prompt", "step8_answer", "step9_selfcheck",
            "final_answer"
        ]}

    result = {}

    # ========== Step 1: 意图识别 ==========
    t0 = time.time()
    intent = _judge_intent(query)
    t1 = time.time()
    result["step1_intent"] = (
        f"## Step 1: 意图识别\n\n"
        f"**用户问题**: {query}\n\n"
        f"**识别意图**: `{intent}`\n\n"
        f"**耗时**: {(t1-t0)*1000:.0f}ms\n\n"
        f"---\n\n"
        f"意图类别说明:\n"
        f"- `product_qa`: 产品相关问题 → 走 RAG 检索\n"
        f"- `safety_concern`: 安全担忧 → 走 RAG 检索 (优先安全文档)\n"
        f"- `chitchat`: 闲聊 → 直接回复\n"
        f"- `out_of_scope`: 超出范围 → 引导联系客服\n"
    )

    if intent in ("chitchat", "out_of_scope"):
        result["step2_bm25"] = "## Step 2: BM25 检索\n\n⏭️ 跳过 (意图非产品问题)"
        result["step3_vector"] = "## Step 3: 向量检索\n\n⏭️ 跳过"
        result["step4_rrf"] = "## Step 4: RRF 融合\n\n⏭️ 跳过"
        result["step5_rerank"] = "## Step 5: Rerank 精排\n\n⏭️ 跳过"
        result["step6_relevance"] = "## Step 6: 相关性检查\n\n⏭️ 跳过"
        result["step7_prompt"] = "## Step 7: LLM Prompt\n\n⏭️ 跳过"
        result["step8_answer"] = "## Step 8: 答案生成\n\n⏭️ 跳过"
        result["step9_selfcheck"] = "## Step 9: 自检\n\n⏭️ 跳过"

        if intent == "chitchat":
            llm = get_llm_client()
            resp = llm.chat(query, system_prompt="你是 SporraMom W3 吸奶器的智能客服助手。友好地回应用户。", temperature=0.5)
            result["final_answer"] = f"## 最终回答\n\n{resp.content}"
        else:
            result["final_answer"] = (
                "## 最终回答\n\n"
                "抱歉，这个问题超出了 SporraMom W3 产品的服务范围。\n\n"
                "建议联系售后: **support@sporramom.com**"
            )
        return result

    # ========== Step 2: BM25 关键词检索 ==========
    import jieba
    t0 = time.time()
    tokens = list(jieba.cut(query))
    bm25_results = _retriever.bm25_index.search(query, config.BM25_TOP_K)
    t1 = time.time()

    bm25_lines = [
        "## Step 2: BM25 关键词检索\n\n",
        f"**分词结果**: `{' | '.join(tokens)}`\n\n",
        f"**耗时**: {(t1-t0)*1000:.0f}ms\n\n",
        f"**原理**: 对每个 chunk 计算 BM25 分数 (基于词频和逆文档频率)\n\n",
        "```\n",
        "分数 = Σ IDF(token) × TF(token in chunk) × (k1+1) / (TF + k1×(1-b+b×dl/avgdl))\n",
        "```\n\n",
        "| 排名 | BM25分数 | 章节路径 | 内容预览 |\n",
        "|------|---------|----------|----------|\n",
    ]
    for rank, (idx, score) in enumerate(bm25_results[:8], 1):
        chunk = _retriever.chunks[idx]
        preview = chunk.content[:60].replace("|", "\\|").replace("\n", " ")
        bm25_lines.append(f"| {rank} | {score:.2f} | {chunk.topic} | {preview}... |\n")

    result["step2_bm25"] = "".join(bm25_lines)

    # ========== Step 3: 向量语义检索 ==========
    t0 = time.time()
    vector_results = _retriever.vector_index.search(query, config.VECTOR_TOP_K)
    t1 = time.time()

    # 获取 query embedding 用于展示
    query_emb = _retriever.vector_index.model.encode(
        [query], normalize_embeddings=True
    )[0]

    vector_lines = [
        "## Step 3: 向量语义检索\n\n",
        f"**耗时**: {(t1-t0)*1000:.0f}ms\n\n",
        f"**Embedding 模型**: `{config.EMBEDDING_MODEL}`\n\n",
        f"**向量维度**: {query_emb.shape[0]}\n\n",
        f"**Query 向量 (前20维)**: `{np.array2string(query_emb[:20], precision=4, separator=', ')}`\n\n",
        f"**原理**: 将 query 和所有 chunk 编码为向量，计算余弦相似度\n\n",
        "| 排名 | 余弦相似度 | 章节路径 | 内容预览 |\n",
        "|------|-----------|----------|----------|\n",
    ]
    for rank, (idx, score) in enumerate(vector_results[:8], 1):
        chunk = _retriever.chunks[idx]
        preview = chunk.content[:60].replace("|", "\\|").replace("\n", " ")
        vector_lines.append(f"| {rank} | {score:.4f} | {chunk.topic} | {preview}... |\n")

    result["step3_vector"] = "".join(vector_lines)

    # ========== Step 4: RRF 融合 ==========
    t0 = time.time()
    rrf_results = _retriever._rrf_fusion(bm25_results, vector_results)
    t1 = time.time()

    bm25_rank_map = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}
    vec_rank_map = {idx: rank for rank, (idx, _) in enumerate(vector_results)}

    rrf_lines = [
        "## Step 4: RRF (Reciprocal Rank Fusion) 融合\n\n",
        f"**耗时**: {(t1-t0)*1000:.0f}ms\n\n",
        f"**RRF 参数 k**: {config.RRF_K}\n\n",
        f"**公式**: `score(doc) = Σ 1/(k + rank_i)`\n\n",
        f"**原理**: 不依赖分数绝对值，只看排名。BM25 和向量检索的分数量纲不同，RRF 用排名来融合，更鲁棒。\n\n",
        "| 排名 | RRF分数 | BM25排名 | 向量排名 | 章节路径 |\n",
        "|------|---------|----------|----------|----------|\n",
    ]
    for rank, (idx, score) in enumerate(rrf_results[:10], 1):
        chunk = _retriever.chunks[idx]
        bm25_r = bm25_rank_map.get(idx, -1)
        vec_r = vec_rank_map.get(idx, -1)
        bm25_str = f"#{bm25_r+1}" if bm25_r >= 0 else "未命中"
        vec_str = f"#{vec_r+1}" if vec_r >= 0 else "未命中"
        rrf_lines.append(f"| {rank} | {score:.4f} | {bm25_str} | {vec_str} | {chunk.topic} |\n")

    result["step4_rrf"] = "".join(rrf_lines)

    # ========== Step 5: Rerank 精排 ==========
    t0 = time.time()
    rerank_candidates = rrf_results[:config.BM25_TOP_K + config.VECTOR_TOP_K]
    reranked = _retriever._rerank(query, rerank_candidates, config.RERANK_TOP_N)
    t1 = time.time()

    rerank_lines = [
        "## Step 5: Rerank 精排 (CrossEncoder)\n\n",
        f"**耗时**: {(t1-t0)*1000:.0f}ms\n\n",
        f"**Reranker 模型**: `{config.RERANKER_MODEL}`\n\n",
        f"**原理**: CrossEncoder 将 query 和 doc 拼接输入，交叉注意力打分。\n"
        f"比 Embedding 的双塔模型更精准 (但更慢，所以只对 top-k 候选做)。\n\n",
        f"**候选数**: {len(rerank_candidates)} → 精排后保留: {len(reranked)}\n\n",
        "| 排名 | Rerank分数 | 章节路径 | 内容预览 |\n",
        "|------|-----------|----------|----------|\n",
    ]
    for rank, (idx, score) in enumerate(reranked, 1):
        chunk = _retriever.chunks[idx]
        preview = chunk.content[:60].replace("|", "\\|").replace("\n", " ")
        rerank_lines.append(f"| {rank} | **{score:.4f}** | {chunk.topic} | {preview}... |\n")

    result["step5_rerank"] = "".join(rerank_lines)

    # ========== Step 6: 相关性检查 ==========
    top_score = reranked[0][1] if reranked else 0
    passed = top_score >= config.RELEVANCE_THRESHOLD

    relevance_lines = [
        "## Step 6: 相关性阈值检查\n\n",
        f"**最高 Rerank 分数**: `{top_score:.4f}`\n\n",
        f"**阈值**: `{config.RELEVANCE_THRESHOLD}`\n\n",
        f"**判定**: {'✅ 通过 — 分数 >= 阈值' if passed else '⚠️ 低于阈值 — 触发降级策略'}\n\n",
    ]

    if not passed:
        relevance_lines.append(
            "**降级策略**:\n"
            "1. 扩大检索范围 (top_k × 2)\n"
            "2. 若仍不足且分数 > 0.01 → 低置信度回答 (加'仅供参考'提示)\n"
            "3. 若分数极低 → 返回'建议联系客服'\n\n"
        )
        # 扩大检索
        expanded = _retriever.retrieve(
            query,
            bm25_top_k=config.BM25_TOP_K * 2,
            vector_top_k=config.VECTOR_TOP_K * 2,
            rerank_top_n=config.RERANK_TOP_N,
        )
        if expanded and expanded[0].rerank_score >= config.RELEVANCE_THRESHOLD * 0.8:
            relevance_lines.append(f"**扩大检索后最高分**: {expanded[0].rerank_score:.4f} ✅ 使用扩大后的结果\n")
            reranked = [(r.chunk.chunk_id, r.rerank_score) for r in expanded]
        elif expanded and expanded[0].rerank_score > 0.01:
            relevance_lines.append(f"**扩大检索后最高分**: {expanded[0].rerank_score:.4f} → 低置信度回答\n")
        else:
            relevance_lines.append(f"**扩大检索后最高分**: {expanded[0].rerank_score if expanded else 0:.4f} → 无法回答\n")

    result["step6_relevance"] = "".join(relevance_lines)

    # ========== Step 7: 构建 Prompt ==========
    # 用 reranked 结果构建上下文
    final_results = _retriever.retrieve(
        query,
        bm25_top_k=config.BM25_TOP_K,
        vector_top_k=config.VECTOR_TOP_K,
        rerank_top_n=config.RERANK_TOP_N,
    )
    context = _format_context(final_results)

    prompt_text = ANSWER_PROMPT.format(context=context, query=query)

    prompt_lines = [
        "## Step 7: LLM Prompt 构建\n\n",
        f"**System Prompt**:\n```\n{SYSTEM_PROMPT[:500]}...\n```\n\n",
        f"**User Prompt (完整)**:\n\n",
        f"```\n{prompt_text}\n```\n\n",
        f"**上下文 chunk 数**: {len(final_results)}\n\n",
        f"**上下文总字符数**: {len(context)}\n\n",
        f"**LLM 参数**: temperature={config.LLM_TEMPERATURE}, max_tokens={config.LLM_MAX_TOKENS}\n",
    ]

    result["step7_prompt"] = "".join(prompt_lines)

    # ========== Step 8: 答案生成 ==========
    t0 = time.time()
    llm_resp = _generate_answer(query, context)
    t1 = time.time()

    answer_lines = [
        "## Step 8: LLM 答案生成\n\n",
        f"**耗时**: {(t1-t0)*1000:.0f}ms\n\n",
        f"**输入 Token**: {llm_resp.input_tokens}\n\n",
        f"**输出 Token**: {llm_resp.output_tokens}\n\n",
        f"**模型**: {llm_resp.model}\n\n",
        f"**生成的答案**:\n\n",
        f"{llm_resp.content}\n",
    ]

    result["step8_answer"] = "".join(answer_lines)

    # ========== Step 9: 自检 ==========
    coverage = _self_check(llm_resp.content, context)

    selfcheck_lines = [
        "## Step 9: 答案自检\n\n",
        f"**关键词覆盖率**: `{coverage:.1%}`\n\n",
        f"**阈值**: `{config.SELF_CHECK_MIN_COVERAGE:.1%}`\n\n",
        f"**判定**: {'✅ 通过' if coverage >= config.SELF_CHECK_MIN_COVERAGE else '⚠️ 覆盖率偏低，用低温度重试'}\n\n",
        f"**原理**: 用 jieba 提取答案中的关键词，检查是否在源文档中出现。\n"
        f"覆盖率低说明答案可能包含编造信息。\n",
    ]

    if coverage < config.SELF_CHECK_MIN_COVERAGE:
        retry_resp = _generate_answer(query, context, temperature=0.0)
        retry_coverage = _self_check(retry_resp.content, context)
        selfcheck_lines.append(f"\n**重试 (temperature=0.0)**: 覆盖率 = {retry_coverage:.1%}\n")
        if retry_coverage > coverage:
            llm_resp = retry_resp
            coverage = retry_coverage
            selfcheck_lines.append("✅ 重试结果更好，使用重试的答案\n")

    result["step9_selfcheck"] = "".join(selfcheck_lines)

    # ========== 最终回答 ==========
    result["final_answer"] = f"## 最终回答\n\n{llm_resp.content}"

    return result


# ============================================================
# 向量数据库查看
# ============================================================

def view_vector_db():
    """返回向量数据库中所有 chunk 的信息"""
    if _retriever is None or _chunks is None:
        return "系统未初始化"

    lines = [
        "# 📦 向量数据库内容\n\n",
        f"**Chunk 总数**: {len(_chunks)}\n\n",
        f"**Embedding 模型**: `{config.EMBEDDING_MODEL}`\n\n",
        f"**向量维度**: {_retriever.vector_index.embeddings.shape[1]}\n\n",
        f"**索引类型**: FAISS IndexFlatIP (内积 = 归一化后的余弦相似度)\n\n",
        "---\n\n",
    ]

    for i, chunk in enumerate(_chunks):
        emb = _retriever.vector_index.embeddings[i]
        lines.append(f"## Chunk {i+1}: {chunk.topic}\n\n")
        lines.append(f"**Chunk ID**: `{chunk.chunk_id}`\n\n")
        lines.append(f"**字符数**: {chunk.char_count}\n\n")
        lines.append(f"**向量 (前10维)**: `{np.array2string(emb[:10], precision=4, separator=', ')}`\n\n")
        lines.append(f"**内容**:\n\n")
        lines.append(f"{chunk.content}\n\n")
        lines.append("---\n\n")

    return "".join(lines)


def search_vector_db(query: str):
    """在向量数据库中搜索，展示 query 与每个 chunk 的相似度"""
    if not query.strip():
        return "请输入查询"

    if _retriever is None or _chunks is None:
        return "系统未初始化"

    query_emb = _retriever.vector_index.model.encode(
        [query], normalize_embeddings=True
    )[0]

    # 计算与所有 chunk 的相似度
    all_sims = []
    for i, chunk in enumerate(_chunks):
        chunk_emb = _retriever.vector_index.embeddings[i]
        sim = float(np.dot(query_emb, chunk_emb))
        all_sims.append((i, sim, chunk))

    # 按相似度排序
    all_sims.sort(key=lambda x: x[1], reverse=True)

    lines = [
        f"# 🔍 向量搜索: \"{query}\"\n\n",
        f"**Query 向量 (前10维)**: `{np.array2string(query_emb[:10], precision=4, separator=', ')}`\n\n",
        "| 排名 | 余弦相似度 | 章节路径 | 内容预览 |\n",
        "|------|-----------|----------|----------|\n",
    ]

    for rank, (idx, sim, chunk) in enumerate(all_sims, 1):
        preview = chunk.content[:80].replace("|", "\\|").replace("\n", " ")
        highlight = "**" if rank <= 3 else ""
        lines.append(f"| {highlight}{rank}{highlight} | {highlight}{sim:.4f}{highlight} | {chunk.topic} | {preview}... |\n")

    lines.append(f"\n**所有 {len(all_sims)} 个 chunk 的相似度分布**:\n\n")
    sims_values = [s[1] for s in all_sims]
    lines.append(f"- 最高: {max(sims_values):.4f}\n")
    lines.append(f"- 最低: {min(sims_values):.4f}\n")
    lines.append(f"- 平均: {np.mean(sims_values):.4f}\n")
    lines.append(f"- 中位数: {np.median(sims_values):.4f}\n")

    return "".join(lines)


# ============================================================
# 构建 Gradio 界面
# ============================================================

def build_ui() -> gr.Blocks:

    with gr.Blocks(title="SporraMom W3 RAG 调试") as app:

        gr.Markdown("# 🍼 SporraMom W3 RAG 全链路调试")

        with gr.Tabs():

            # ====== Tab 1: 全链路调试 ======
            with gr.Tab("🔬 全链路调试"):
                gr.Markdown("输入一个问题，查看 RAG 流程中每一步的中间数据。")

                with gr.Row():
                    query_input = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：吸奶器充不了电怎么办？",
                        scale=4,
                    )
                    run_btn = gr.Button("🚀 运行", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "我们的产品是啥",
                        "吸奶器充不了电怎么办",
                        "母乳可以保存多久",
                        "喇叭罩尺寸怎么选",
                        "使用时感到疼痛正常吗",
                        "你好呀",
                        "iPhone 16 多少钱",
                    ],
                    inputs=query_input,
                )

                # 最终回答放在最上面
                final_answer = gr.Markdown(label="最终回答")

                with gr.Accordion("Step 1: 意图识别", open=True):
                    step1 = gr.Markdown()

                with gr.Accordion("Step 2: BM25 关键词检索", open=True):
                    step2 = gr.Markdown()

                with gr.Accordion("Step 3: 向量语义检索", open=True):
                    step3 = gr.Markdown()

                with gr.Accordion("Step 4: RRF 融合", open=True):
                    step4 = gr.Markdown()

                with gr.Accordion("Step 5: Rerank 精排", open=True):
                    step5 = gr.Markdown()

                with gr.Accordion("Step 6: 相关性检查", open=True):
                    step6 = gr.Markdown()

                with gr.Accordion("Step 7: LLM Prompt", open=True):
                    step7 = gr.Markdown()

                with gr.Accordion("Step 8: 答案生成", open=True):
                    step8 = gr.Markdown()

                with gr.Accordion("Step 9: 自检", open=True):
                    step9 = gr.Markdown()

                def run_debug(query):
                    results = debug_full_pipeline(query)
                    return (
                        results["final_answer"],
                        results["step1_intent"],
                        results["step2_bm25"],
                        results["step3_vector"],
                        results["step4_rrf"],
                        results["step5_rerank"],
                        results["step6_relevance"],
                        results["step7_prompt"],
                        results["step8_answer"],
                        results["step9_selfcheck"],
                    )

                run_btn.click(
                    run_debug,
                    inputs=query_input,
                    outputs=[
                        final_answer, step1, step2, step3, step4,
                        step5, step6, step7, step8, step9,
                    ],
                )
                query_input.submit(
                    run_debug,
                    inputs=query_input,
                    outputs=[
                        final_answer, step1, step2, step3, step4,
                        step5, step6, step7, step8, step9,
                    ],
                )

            # ====== Tab 2: 向量数据库 ======
            with gr.Tab("📦 向量数据库"):
                gr.Markdown("查看向量数据库中所有 chunk 的内容和 embedding 向量。")

                with gr.Row():
                    db_btn = gr.Button("📋 加载全部 Chunk", variant="primary")

                db_output = gr.Markdown()

                gr.Markdown("---")
                gr.Markdown("### 🔍 向量搜索测试")
                gr.Markdown("输入查询，查看它与每个 chunk 的余弦相似度。")

                with gr.Row():
                    search_input = gr.Textbox(
                        label="搜索查询",
                        placeholder="输入任意文本...",
                        scale=4,
                    )
                    search_btn = gr.Button("搜索", variant="primary", scale=1)

                search_output = gr.Markdown()

                db_btn.click(view_vector_db, outputs=db_output)
                search_btn.click(search_vector_db, inputs=search_input, outputs=search_output)
                search_input.submit(search_vector_db, inputs=search_input, outputs=search_output)

            # ====== Tab 3: 评测 ======
            with gr.Tab("📊 评测"):
                gr.Markdown("一键运行评测，查看指标。")

                eval_btn = gr.Button("🚀 开始评测", variant="primary")
                eval_output = gr.Markdown()

                def run_eval_display():
                    from evaluator import run_evaluation, format_eval_report, generate_test_set, save_test_set
                    save_test_set()
                    test_cases = generate_test_set()
                    report = run_evaluation(_retriever, test_cases, eval_relevance=False)
                    return format_eval_report(report)

                eval_btn.click(run_eval_display, outputs=eval_output)

    return app


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    summary = init_system()

    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
