"""
SporraMom W3 智能客服系统 — RAG 主流程

设计思路（面试要点 — 工作流稳定性）:
  三道防线确保回答质量和系统稳健性:

  防线1: 意图过滤
    - 用户问的是闲聊、超出范围的问题 → 直接礼貌回复，不走检索
    - 避免无关问题污染检索结果

  防线2: 相关性阈值检查
    - Rerank 最高分 < threshold → 说明知识库中没有相关内容
    - 触发降级策略: 扩大检索 → 放宽阈值 → 转人工客服
    - 避免"硬答"导致幻觉

  防线3: 答案自检
    - 生成的答案与源文档做关键词覆盖率检查
    - 覆盖率低 → 用更低温度重新生成 (提高忠实度)
    - 避免模型编造不在文档中的信息
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import config
from data_processor import Chunk
from retriever import HybridRetriever, RetrievalResult
from llm_client import get_llm_client, LLMResponse


@dataclass
class RAGResponse:
    """RAG 流程完整响应"""
    answer: str                          # 最终回答
    sources: List[Dict[str, Any]]        # 引用来源
    retrieval_results: List[Dict]        # 检索详情 (用于可视化)
    intent: str                          # 识别的意图
    fallback_triggered: bool = False     # 是否触发了降级策略
    self_check_passed: bool = True       # 自检是否通过
    llm_response: Optional[LLMResponse] = None


# 系统提示词
SYSTEM_PROMPT = """你是 SporraMom W3 穿戴式吸奶器的智能客服助手。

你的职责:
1. 基于提供的产品文档，准确回答用户关于 SporraMom W3 的问题
2. 回答必须忠实于文档内容，不要编造信息
3. 如果文档中没有相关信息，明确告知用户并建议联系售后
4. 回答要简洁、专业、有条理
5. 涉及安全问题时，优先引用安全警告内容
6. 如果用户问的是产品概述类问题（如"产品是啥""介绍一下产品"），基于文档中的适用范围和产品说明来回答

回答格式要求:
- 使用中文回答
- 重要信息用加粗标注
- 涉及步骤时使用有序列表
- 涉及安全警告时明确标注 ⚠️"""

# 意图判断提示词
INTENT_PROMPT = """判断用户问题的意图类别，只返回类别名称，不要其他内容。

类别:
- product_qa: 关于产品使用、功能、参数、故障排除的问题
- safety_concern: 关于安全、禁忌、副作用的担忧
- chitchat: 闲聊、打招呼、感谢
- out_of_scope: 超出产品范围的问题 (如其他品牌、医疗诊断等)

用户问题: {query}
类别:"""

# 答案生成提示词
ANSWER_PROMPT = """基于以下产品文档内容，回答用户问题。

要求:
1. 只使用提供的文档内容回答，不要编造
2. 如果文档不足以回答，明确说明
3. 回答要准确、简洁

相关文档内容:
{context}

用户问题: {query}

回答:"""


def _judge_intent(query: str) -> str:
    """判断用户意图"""
    llm = get_llm_client()
    try:
        resp = llm.chat(
            INTENT_PROMPT.format(query=query),
            temperature=0.1,
            max_tokens=20,
        )
        intent = resp.content.strip().lower()
        valid_intents = ["product_qa", "safety_concern", "chitchat", "out_of_scope"]
        for v in valid_intents:
            if v in intent:
                return v
        return "product_qa"  # 默认当产品问题处理
    except Exception:
        return "product_qa"  # 出错时默认处理


def _check_relevance(results: List[RetrievalResult], threshold: float) -> bool:
    """检查检索结果的相关性是否达标"""
    if not results:
        return False
    return results[0].rerank_score >= threshold


def _format_context(results: List[RetrievalResult]) -> str:
    """将检索结果格式化为 LLM 上下文"""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[文档{i}] (来源: {r.chunk.topic})\n{r.chunk.content}")
    return "\n\n".join(parts)


def _self_check(answer: str, context: str) -> float:
    """
    自检: 检查答案中的关键信息是否在源文档中有支撑。

    简单实现: 用 jieba 提取关键词，计算覆盖率。
    更高级: 可以用 NLI 模型判断蕴含关系。
    """
    import jieba

    # 提取答案中的关键词 (去掉停用词)
    stop_words = {"的", "了", "是", "在", "和", "有", "为", "这", "个", "我", "你", "他",
                  "她", "它", "们", "那", "就", "也", "都", "而", "及", "与", "或", "但",
                  "如果", "因为", "所以", "可以", "请", "需要", "不", "没有", "已经",
                  "将", "被", "从", "到", "对", "等", "呢", "吗", "啊", "哦", "嗯",
                  "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}

    answer_words = set(jieba.cut(answer)) - stop_words
    answer_words = {w for w in answer_words if len(w) > 1}  # 只保留多字词

    if not answer_words:
        return 1.0

    # 检查关键词在上下文中的出现
    matched = sum(1 for w in answer_words if w in context)
    coverage = matched / len(answer_words)
    return coverage


def _generate_answer(query: str, context: str, temperature: float = None) -> LLMResponse:
    """调用 LLM 生成答案"""
    llm = get_llm_client()
    return llm.chat(
        ANSWER_PROMPT.format(context=context, query=query),
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
    )


def _handle_chitchat(query: str) -> RAGResponse:
    """处理闲聊"""
    llm = get_llm_client()
    resp = llm.chat(
        query,
        system_prompt="你是 SporraMom W3 吸奶器的智能客服助手。友好地回应用户，如果是闲聊可以简短回复，然后引导用户询问产品相关问题。",
        temperature=0.5,
    )
    return RAGResponse(
        answer=resp.content,
        sources=[],
        retrieval_results=[],
        intent="chitchat",
        llm_response=resp,
    )


def _handle_out_of_scope(query: str) -> RAGResponse:
    """处理超出范围的问题"""
    return RAGResponse(
        answer="抱歉，这个问题超出了 SporraMom W3 产品的服务范围。建议您：\n\n"
               "1. 联系我们的售后邮箱：**support@sporramom.com**\n"
               "2. 访问官网获取更多信息：**https://www.sporramom.com/**\n\n"
               "如果您有关于 SporraMom W3 吸奶器的使用问题，我很乐意为您解答！",
        sources=[],
        retrieval_results=[],
        intent="out_of_scope",
    )


def _fallback_response(query: str) -> RAGResponse:
    """降级策略: 当知识库中没有相关内容时的回复"""
    return RAGResponse(
        answer="抱歉，我在产品文档中没有找到与您问题直接相关的信息。\n\n"
               "建议您：\n"
               "1. **换个方式描述您的问题**，可能关键词不同\n"
               "2. **联系售后客服**：support@sporramom.com\n"
               "3. **查看故障排除章节**，也许有类似问题的解决方案\n\n"
               "您可以再试试其他关于 SporraMom W3 的问题！",
        sources=[],
        retrieval_results=[],
        intent="product_qa",
        fallback_triggered=True,
    )


def run_rag(
    query: str,
    retriever: HybridRetriever,
    bm25_top_k: int = None,
    vector_top_k: int = None,
    rerank_top_n: int = None,
    relevance_threshold: float = None,
) -> RAGResponse:
    """
    完整 RAG 流程。

    流程:
    1. 意图判断 → 闲聊/超范围直接回复
    2. 混合检索 → BM25 + 向量 + RRF + Rerank
    3. 相关性检查 → 低于阈值触发降级
    4. 答案生成 → LLM 基于检索结果生成
    5. 自检 → 覆盖率低则重试
    """
    relevance_threshold = relevance_threshold or config.RELEVANCE_THRESHOLD

    # Step 1: 意图判断
    intent = _judge_intent(query)

    if intent == "chitchat":
        return _handle_chitchat(query)
    if intent == "out_of_scope":
        return _handle_out_of_scope(query)

    # Step 2: 混合检索
    results = retriever.retrieve(
        query,
        bm25_top_k=bm25_top_k,
        vector_top_k=vector_top_k,
        rerank_top_n=rerank_top_n,
    )

    # 检索详情 (用于可视化)
    retrieval_details = []
    for r in results:
        retrieval_details.append({
            "chunk_id": r.chunk.chunk_id,
            "topic": r.chunk.topic,
            "content_preview": r.chunk.content[:150],
            "bm25_rank": r.bm25_rank,
            "vector_rank": r.vector_rank,
            "rrf_score": round(r.rrf_score, 4),
            "rerank_score": round(r.rerank_score, 4),
        })

    # Step 3: 相关性检查 + 降级策略
    low_confidence = False
    if not _check_relevance(results, relevance_threshold):
        # 降级策略1: 扩大检索范围
        expanded_results = retriever.retrieve(
            query,
            bm25_top_k=(bm25_top_k or config.BM25_TOP_K) * 2,
            vector_top_k=(vector_top_k or config.VECTOR_TOP_K) * 2,
            rerank_top_n=rerank_top_n,
        )

        if _check_relevance(expanded_results, relevance_threshold * 0.8):
            # 扩大后找到了，用扩大后的结果
            results = expanded_results
            for r in results:
                retrieval_details.append({
                    "chunk_id": r.chunk.chunk_id,
                    "topic": r.chunk.topic,
                    "content_preview": r.chunk.content[:150],
                    "bm25_rank": r.bm25_rank,
                    "vector_rank": r.vector_rank,
                    "rrf_score": round(r.rrf_score, 4),
                    "rerank_score": round(r.rerank_score, 4),
                    "note": "expanded_search",
                })
        elif results and results[0].rerank_score > 0.01:
            # 降级策略2: 有检索结果但置信度低 → 尝试生成，标记低置信度
            low_confidence = True
        else:
            # 降级策略3: 几乎无相关结果 → 返回降级回复
            return _fallback_response(query)

    # Step 4: 答案生成
    context = _format_context(results)
    if low_confidence:
        # 低置信度时，提示 LLM 更谨慎地回答
        llm_resp = _generate_answer(query, context, temperature=0.1)
        # 在答案前加提示
        llm_resp = LLMResponse(
            content="💡 以下回答基于有限的检索结果，仅供参考：\n\n" + llm_resp.content,
            input_tokens=llm_resp.input_tokens,
            output_tokens=llm_resp.output_tokens,
            latency_ms=llm_resp.latency_ms,
            model=llm_resp.model,
        )
    else:
        llm_resp = _generate_answer(query, context)

    # Step 5: 自检
    coverage = _self_check(llm_resp.content, context)
    self_check_passed = True

    if coverage < config.SELF_CHECK_MIN_COVERAGE:
        # 自检失败: 用更低温度重试一次
        llm_resp_retry = _generate_answer(query, context, temperature=0.0)
        coverage_retry = _self_check(llm_resp_retry.content, context)

        if coverage_retry > coverage:
            llm_resp = llm_resp_retry
            coverage = coverage_retry

        if coverage < config.SELF_CHECK_MIN_COVERAGE:
            self_check_passed = False

    # 组装来源信息
    sources = []
    for r in results:
        sources.append({
            "section": r.chunk.topic,
            "content": r.chunk.content[:200],
            "rerank_score": round(r.rerank_score, 3),
        })

    return RAGResponse(
        answer=llm_resp.content,
        sources=sources,
        retrieval_results=retrieval_details,
        intent=intent,
        fallback_triggered=False,
        self_check_passed=self_check_passed,
        llm_response=llm_resp,
    )


if __name__ == "__main__":
    from data_processor import load_or_build_chunks
    from retriever import build_retriever

    chunks = load_or_build_chunks()
    retriever = build_retriever(chunks)

    test_queries = [
        "你好呀！",
        "吸奶器充不了电怎么办",
        "母乳可以保存多久？",
        "我想买个 iPhone",
        "这个吸奶器和美德乐比哪个好",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        resp = run_rag(q, retriever)
        print(f"意图: {resp.intent} | 降级: {resp.fallback_triggered} | 自检: {resp.self_check_passed}")
        print(f"A: {resp.answer[:200]}...")
