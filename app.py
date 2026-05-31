"""
SporraMom W3 智能客服系统 — Gradio 界面

界面设计:
  Tab 1: 智能客服聊天 — 核心功能
  Tab 2: 检索过程可视化 — 展示 RAG 内部工作原理
  Tab 3: 模型评测 — 一键运行评测，展示指标
  Tab 4: 系统配置 — 可调参数
"""

import time
import json
from typing import List, Tuple, Dict, Any

import gradio as gr

import config
from data_processor import load_or_build_chunks, get_chunks_summary
from retriever import HybridRetriever, build_retriever
from rag_pipeline import run_rag, RAGResponse
from evaluator import (
    generate_test_set, run_evaluation, format_eval_report,
    save_test_set, EvalReport
)

# ============================================================
# 全局状态
# ============================================================
_retriever: HybridRetriever = None
_eval_report: EvalReport = None


def init_system():
    """初始化系统: 加载数据、构建索引"""
    global _retriever

    print("=" * 60)
    print("SporraMom W3 智能客服系统 — 初始化中...")
    print("=" * 60)

    # 加载数据
    chunks = load_or_build_chunks()
    summary = get_chunks_summary(chunks)
    print(f"数据摘要: {summary['total_chunks']} 个 chunks, {summary['total_chars']} 字符")

    # 构建索引
    _retriever = build_retriever(chunks)

    # 生成测试集
    save_test_set()

    print("=" * 60)
    print("初始化完成！")
    print("=" * 60)

    return summary


# ============================================================
# Tab 1: 智能客服
# ============================================================

def chat_respond(message: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    处理用户消息，返回回答和检索详情。
    """
    if not message.strip():
        return "", "请输入问题"

    if _retriever is None:
        return "系统未初始化，请稍候...", ""

    # 执行 RAG
    start_time = time.time()
    response = run_rag(message, _retriever)
    total_time = int((time.time() - start_time) * 1000)

    # 构建检索详情
    details = []
    details.append(f"🔍 意图识别: {response.intent}")
    details.append(f"⏱️ 响应时间: {total_time}ms")
    details.append(f"🔄 降级触发: {'是' if response.fallback_triggered else '否'}")
    details.append(f"✅ 自检通过: {'是' if response.self_check_passed else '否'}")
    details.append("")

    if response.retrieval_results:
        details.append("📄 检索结果:")
        for i, r in enumerate(response.retrieval_results[:5], 1):
            details.append(f"  [{i}] Rerank={r['rerank_score']:.3f} | {r['section_path']}")
            details.append(f"      {r['content_preview'][:80]}...")
            details.append("")

    if response.sources:
        details.append("📚 引用来源:")
        for s in response.sources[:3]:
            details.append(f"  • {s['section']} (相关度: {s['rerank_score']:.3f})")

    return response.answer, "\n".join(details)


# ============================================================
# Tab 2: 检索过程可视化
# ============================================================

def visualize_retrieval(query: str) -> str:
    """展示检索的完整过程"""
    if not query.strip():
        return "请输入查询"

    if _retriever is None:
        return "系统未初始化"

    lines = []
    lines.append(f"查询: {query}")
    lines.append("=" * 60)

    # BM25 检索
    import jieba
    tokens = list(jieba.cut(query))
    lines.append(f"\n📝 分词结果: {' | '.join(tokens)}")

    bm25_results = _retriever.bm25_index.search(query, config.BM25_TOP_K)
    lines.append(f"\n🔑 BM25 关键词检索 (Top {len(bm25_results)}):")
    for rank, (idx, score) in enumerate(bm25_results[:5], 1):
        chunk = _retriever.chunks[idx]
        lines.append(f"  [{rank}] score={score:.2f} | {chunk.section_path}")
        lines.append(f"      {chunk.content[:80]}...")

    # 向量检索
    vector_results = _retriever.vector_index.search(query, config.VECTOR_TOP_K)
    lines.append(f"\n🧠 向量语义检索 (Top {len(vector_results)}):")
    for rank, (idx, score) in enumerate(vector_results[:5], 1):
        chunk = _retriever.chunks[idx]
        lines.append(f"  [{rank}] cos_sim={score:.4f} | {chunk.section_path}")
        lines.append(f"      {chunk.content[:80]}...")

    # RRF 融合
    rrf_results = _retriever._rrf_fusion(bm25_results, vector_results)
    lines.append(f"\n🔀 RRF 融合 (Top 5):")
    for rank, (idx, score) in enumerate(rrf_results[:5], 1):
        chunk = _retriever.chunks[idx]
        bm25_rank = next((r for i, (i2, _) in enumerate(bm25_results) if i2 == idx), -1)
        vec_rank = next((r for i, (i2, _) in enumerate(vector_results) if i2 == idx), -1)
        lines.append(f"  [{rank}] rrf={score:.4f} | BM25#{bm25_rank+1 if bm25_rank >= 0 else 'N/A'} Vec#{vec_rank+1 if vec_rank >= 0 else 'N/A'} | {chunk.section_path}")

    # Rerank
    rerank_candidates = rrf_results[:config.BM25_TOP_K + config.VECTOR_TOP_K]
    reranked = _retriever._rerank(query, rerank_candidates, config.RERANK_TOP_N)
    lines.append(f"\n🎯 Rerank 精排 (Top {len(reranked)}):")
    for rank, (idx, score) in enumerate(reranked, 1):
        chunk = _retriever.chunks[idx]
        lines.append(f"  [{rank}] rerank_score={score:.4f} | {chunk.section_path}")
        lines.append(f"      {chunk.content[:100]}...")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Tab 3: 模型评测
# ============================================================

def run_eval(
    eval_relevance: bool = False,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """运行评测并返回报告"""
    global _eval_report

    if _retriever is None:
        return "系统未初始化", ""

    test_cases = generate_test_set()

    # 简化: 不用 progress 因为 run_evaluation 内部已有打印
    _eval_report = run_evaluation(_retriever, test_cases, eval_relevance=eval_relevance)

    report_text = format_eval_report(_eval_report)

    # JSON 详情
    details = []
    for r in _eval_report.details:
        details.append({
            "query": r.query,
            "category": r.category,
            "hit_rate@3": r.hit_rate_at_3,
            "mrr": round(r.mrr, 3),
            "keyword_coverage": round(r.keyword_coverage, 3),
            "response_time_ms": r.response_time_ms,
            "fallback": r.fallback_triggered,
        })
    details_json = json.dumps(details, ensure_ascii=False, indent=2)

    return report_text, details_json


# ============================================================
# Tab 4: 系统配置
# ============================================================

def get_current_config() -> str:
    """获取当前配置"""
    lines = []
    lines.append("=== 检索参数 ===")
    lines.append(f"BM25 Top K:       {config.BM25_TOP_K}")
    lines.append(f"Vector Top K:     {config.VECTOR_TOP_K}")
    lines.append(f"RRF K:            {config.RRF_K}")
    lines.append(f"Rerank Top N:     {config.RERANK_TOP_N}")
    lines.append(f"Relevance Thresh: {config.RELEVANCE_THRESHOLD}")
    lines.append("")
    lines.append("=== 分块参数 ===")
    lines.append(f"Chunk Max Chars:  {config.CHUNK_MAX_CHARS}")
    lines.append(f"Chunk Overlap:    {config.CHUNK_OVERLAP}")
    lines.append("")
    lines.append("=== 模型参数 ===")
    lines.append(f"Embedding Model:  {config.EMBEDDING_MODEL}")
    lines.append(f"Reranker Model:   {config.RERANKER_MODEL}")
    lines.append(f"LLM Model:        {config.LLM_MODEL}")
    lines.append(f"Temperature:      {config.LLM_TEMPERATURE}")
    lines.append(f"Max Tokens:       {config.LLM_MAX_TOKENS}")
    lines.append("")
    lines.append("=== 自检参数 ===")
    lines.append(f"Min Coverage:     {config.SELF_CHECK_MIN_COVERAGE}")
    lines.append(f"Max Retries:      {config.SELF_CHECK_MAX_RETRIES}")

    return "\n".join(lines)


# ============================================================
# 构建 Gradio 界面
# ============================================================

def build_ui() -> gr.Blocks:
    """构建 Gradio 界面"""

    with gr.Blocks(
        title="SporraMom W3 智能客服系统",
    ) as app:

        gr.Markdown("""
        # 🍼 SporraMom W3 智能客服系统
        **RAG 向量检索 + BM25 关键词检索 + Rerank 重排序**

        基于产品说明书构建的智能问答系统，支持语义理解和关键词精确匹配。
        """, elem_classes=["header"])

        with gr.Tabs():

            # ---- Tab 1: 智能客服 ----
            with gr.Tab("💬 智能客服"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=400,
                        )
                        msg_input = gr.Textbox(
                            label="输入问题",
                            placeholder="例如：吸奶器充不了电怎么办？",
                            lines=1,
                        )
                        with gr.Row():
                            submit_btn = gr.Button("发送", variant="primary")
                            clear_btn = gr.Button("清空对话")

                    with gr.Column(scale=1):
                        details_output = gr.Textbox(
                            label="检索过程详情",
                            lines=15,
                            interactive=False,
                        )

                # 示例问题
                gr.Examples(
                    examples=[
                        "吸奶器充不了电怎么办？",
                        "喇叭罩尺寸怎么选？",
                        "母乳可以保存多久？",
                        "使用时感到疼痛正常吗？",
                        "吸奶器怎么组装？",
                        "App 连接失败怎么办？",
                    ],
                    inputs=msg_input,
                )

                def chat_handler(message, history):
                    answer, details = chat_respond(message, history)
                    history = history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": answer},
                    ]
                    return "", history, details

                submit_btn.click(
                    chat_handler,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot, details_output],
                )
                msg_input.submit(
                    chat_handler,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot, details_output],
                )
                clear_btn.click(lambda: ("", None), outputs=[msg_input, chatbot])

            # ---- Tab 2: 检索可视化 ----
            with gr.Tab("🔍 检索可视化"):
                gr.Markdown("### 检索过程可视化\n输入查询，查看 BM25 → 向量 → RRF → Rerank 的完整流程")
                with gr.Row():
                    viz_input = gr.Textbox(
                        label="查询",
                        placeholder="输入任意问题...",
                        scale=3,
                    )
                    viz_btn = gr.Button("检索", variant="primary", scale=1)

                viz_output = gr.Textbox(
                    label="检索过程",
                    lines=25,
                    interactive=False,
                )

                gr.Examples(
                    examples=[
                        "吸奶器没有吸力了",
                        "IP22 防水等级是什么意思",
                        "怎么清洗吸奶器",
                        "电池能用多久",
                    ],
                    inputs=viz_input,
                )

                viz_btn.click(visualize_retrieval, inputs=viz_input, outputs=viz_output)
                viz_input.submit(visualize_retrieval, inputs=viz_input, outputs=viz_output)

            # ---- Tab 3: 模型评测 ----
            with gr.Tab("📊 模型评测"):
                gr.Markdown("""
                ### 评测体系

                **检索质量指标:**
                - **Hit Rate@K**: top-K 结果中是否包含正确文档 (越高越好)
                - **MRR**: 正确文档排名的倒数均值 (越高越好)

                **生成质量指标:**
                - **关键词覆盖率**: 答案中包含期望关键词的比例 (越高越好)

                **系统指标:**
                - **响应时间**: 端到端延迟 (越低越好)
                - **降级触发率**: 触发降级策略的比例 (越低越好)
                - **自检通过率**: 答案自检通过的比例 (越高越好)
                """)

                with gr.Row():
                    eval_relevance = gr.Checkbox(
                        label="启用 LLM 相关性评测 (更准确但更慢)",
                        value=False,
                    )
                    eval_btn = gr.Button("🚀 开始评测", variant="primary")

                eval_report = gr.Textbox(
                    label="评测报告",
                    lines=30,
                    interactive=False,
                )
                eval_details = gr.Textbox(
                    label="评测详情 (JSON)",
                    lines=10,
                    interactive=False,
                )

                eval_btn.click(
                    run_eval,
                    inputs=[eval_relevance],
                    outputs=[eval_report, eval_details],
                )

            # ---- Tab 4: 系统配置 ----
            with gr.Tab("⚙️ 系统配置"):
                gr.Markdown("### 当前配置参数\n修改 config.py 文件后重启生效")

                config_display = gr.Textbox(
                    label="当前配置",
                    value=get_current_config(),
                    lines=25,
                    interactive=False,
                )

                gr.Markdown("""
                ### 配置说明

                | 参数 | 说明 | 调优建议 |
                |------|------|----------|
                | BM25_TOP_K | BM25 初筛候选数 | 增大→召回更多但更慢 |
                | VECTOR_TOP_K | 向量检索候选数 | 增大→召回更多但更慢 |
                | RRF_K | RRF 融合参数 | 推荐 60，论文默认值 |
                | RERANK_TOP_N | Rerank 后保留数 | 增大→上下文更长但可能引入噪声 |
                | RELEVANCE_THRESHOLD | 相关性阈值 | 降低→更宽松，升高→更严格 |
                | CHUNK_MAX_CHARS | 最大 chunk 长度 | 减小→检索更精准，增大→上下文更完整 |
                | LLM_TEMPERATURE | LLM 温度 | 降低→更忠实，升高→更灵活 |

                ### 面试 A/B 实验建议

                1. **检索方式对比**: 关闭向量检索 → 纯 BM25 评测 → 对比混合检索
                2. **分块策略对比**: 调整 CHUNK_MAX_CHARS → 对比 Hit Rate
                3. **温度对比**: 调整 LLM_TEMPERATURE → 对比关键词覆盖率
                """)

        # 底部信息
        gr.Markdown("""
        ---
        **技术栈**: RAG (BM25 + FAISS + BGE Embedding + BGE Reranker) + Gradio

        **模型**: mimo-v2.5-pro (Anthropic 兼容 API)

        **数据**: SporraMom W3 产品说明书 (结构化 Markdown → 语义切分)

        **评测**: Hit Rate / MRR / 关键词覆盖率 / 响应时间 / 降级率
        """)

    return app


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    # 初始化系统
    summary = init_system()

    # 构建并启动 UI
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
