"""
SporraMom W3 智能客服系统 — 模型评测体系

面试核心要点 — "如何证明你的模型效果变好了":

  1. 检索质量指标 (衡量 RAG 检索环节)
     - Hit Rate@K: top-K 结果中是否包含正确文档
       → 业务含义: 用户问的问题，系统能不能"找到"相关文档
     - MRR (Mean Reciprocal Rank): 正确文档排名的倒数均值
       → 业务含义: 相关文档排在第几位，排名越靠前越好

  2. 生成质量指标 (衡量 LLM 生成环节)
     - Faithfulness (忠实度): 答案是否忠实于源文档
       → 业务含义: 有没有编造信息 (幻觉)
     - Relevance (相关性): 答案是否回答了用户问题
       → 业务含义: 答非所问的情况
     - Completeness (完整性): 是否覆盖了问题的所有方面
       → 业务含义: 回答是否全面

  3. 系统指标 (衡量整体流程)
     - Response Time: 端到端延迟
     - Fallback Rate: 触发降级策略的比例
     - Self-check Pass Rate: 自检通过率

  如何用这把"尺子":
  - 调整 chunk_size → 重新评测 → 对比 MRR/Hit Rate
  - A/B 对比: 纯 BM25 vs 纯向量 vs 混合+Rerank
  - 调整 temperature → 重新评测 → 对比 Faithfulness
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import jieba
import numpy as np

import config
from data_processor import Chunk
from retriever import HybridRetriever
from rag_pipeline import run_rag, RAGResponse


# ============================================================
# 测试集定义
# ============================================================

@dataclass
class TestCase:
    """单个测试用例"""
    query: str                           # 用户问题
    expected_keywords: List[str]         # 期望答案包含的关键词
    expected_section: str                # 期望命中的章节 (模糊匹配)
    category: str = "general"            # 测试类别
    difficulty: str = "normal"           # 难度: easy/normal/hard


def generate_test_set() -> List[TestCase]:
    """
    自动生成测试集。

    覆盖维度:
    - 安全警告类 (高优先级)
    - 操作步骤类
    - 故障排除类
    - 技术参数类
    - App 使用类
    - 存储相关类
    - 边界/陷阱问题
    """
    test_cases = [
        # === 安全警告类 ===
        TestCase(
            query="吸奶器可以在充电的时候用吗？",
            expected_keywords=["充电", "禁止", "边充边用", "电气安全"],
            expected_section="安全说明",
            category="safety",
            difficulty="easy",
        ),
        TestCase(
            query="孕期可以用这个吸奶器吗？",
            expected_keywords=["孕期", "哺乳期", "宫缩", "提前临产"],
            expected_section="安全说明",
            category="safety",
            difficulty="easy",
        ),
        TestCase(
            query="使用时感到疼痛正常吗？",
            expected_keywords=["疼痛", "立即停止", "乳头", "损伤"],
            expected_section="安全说明",
            category="safety",
            difficulty="normal",
        ),
        TestCase(
            query="吸奶器可以给别人用吗？",
            expected_keywords=["单人专用", "个人卫生", "交叉污染", "共用"],
            expected_section="安全说明",
            category="safety",
            difficulty="easy",
        ),
        TestCase(
            query="睡觉的时候可以戴着吸奶器吗？",
            expected_keywords=["睡眠", "禁止", "困倦", "无人注意"],
            expected_section="安全说明",
            category="safety",
            difficulty="normal",
        ),

        # === 操作步骤类 ===
        TestCase(
            query="吸奶器怎么组装？",
            expected_keywords=["集乳杯", "鸭嘴阀", "硅胶隔膜", "卡位", "咔嗒"],
            expected_section="组装",
            category="operation",
            difficulty="normal",
        ),
        TestCase(
            query="怎么开始吸奶？",
            expected_keywords=["电源键", "刺激模式", "吸乳模式", "对准", "喇叭罩"],
            expected_section="吸取母乳",
            category="operation",
            difficulty="normal",
        ),
        TestCase(
            query="吸奶器的按钮怎么用？",
            expected_keywords=["电源键", "切换键", "加档", "减档", "长按", "短按"],
            expected_section="产品说明",
            category="operation",
            difficulty="easy",
        ),

        # === 故障排除类 ===
        TestCase(
            query="吸奶器开不了机怎么办？",
            expected_keywords=["电池耗尽", "充电", "长按", "电源键", "1.5秒"],
            expected_section="故障排除",
            category="troubleshoot",
            difficulty="normal",
        ),
        TestCase(
            query="吸奶器没有吸力了是什么原因？",
            expected_keywords=["组装", "隔膜", "鸭嘴阀", "密封"],
            expected_section="故障排除",
            category="troubleshoot",
            difficulty="normal",
        ),
        TestCase(
            query="吸力太大了很痛怎么办？",
            expected_keywords=["档位", "调低", "尺寸", "硅胶塞"],
            expected_section="故障排除",
            category="troubleshoot",
            difficulty="normal",
        ),
        TestCase(
            query="乳汁倒流进主机了怎么办？",
            expected_keywords=["隔膜", "破损", "移位", "停止使用", "售后"],
            expected_section="故障排除",
            category="troubleshoot",
            difficulty="hard",
        ),

        # === 技术参数类 ===
        TestCase(
            query="电池容量是多少？",
            expected_keywords=["1400mAh", "锂离子", "3.7V"],
            expected_section="技术规格",
            category="specs",
            difficulty="easy",
        ),
        TestCase(
            query="充满电要多久？",
            expected_keywords=["2.5小时", "充电"],
            expected_section="技术规格",
            category="specs",
            difficulty="easy",
        ),
        TestCase(
            query="噪音大吗？",
            expected_keywords=["50dB", "噪音"],
            expected_section="技术规格",
            category="specs",
            difficulty="easy",
        ),
        TestCase(
            query="防水等级是多少？",
            expected_keywords=["IP22", "防护等级", "不可水洗"],
            expected_section="技术规格",
            category="specs",
            difficulty="easy",
        ),

        # === 尺寸选择类 ===
        TestCase(
            query="喇叭罩尺寸怎么选？",
            expected_keywords=["乳头", "直径", "18mm", "20mm", "22mm", "1~3mm"],
            expected_section="选择正确的吸乳护罩尺寸",
            category="sizing",
            difficulty="normal",
        ),
        TestCase(
            query="硅胶塞太大了会怎样？",
            expected_keywords=["乳晕", "被吸入", "乳腺管", "出奶少"],
            expected_section="选择正确的吸乳护罩尺寸",
            category="sizing",
            difficulty="normal",
        ),

        # === 清洁消毒类 ===
        TestCase(
            query="吸奶器怎么清洗？",
            expected_keywords=["触奶部件", "温水", "母婴清洗剂", "煮沸消毒", "晾干"],
            expected_section="清洁与消杀",
            category="cleaning",
            difficulty="normal",
        ),
        TestCase(
            query="主机可以水洗吗？",
            expected_keywords=["不可水洗", "IP22", "湿布", "擦拭"],
            expected_section="清洁与消杀",
            category="cleaning",
            difficulty="easy",
        ),

        # === 存储类 ===
        TestCase(
            query="母乳可以保存多久？",
            expected_keywords=["常温", "4小时", "冷藏", "3-5天", "冷冻", "6个月"],
            expected_section="倒奶与存奶",
            category="storage",
            difficulty="normal",
        ),

        # === App 相关 ===
        TestCase(
            query="App 怎么连接吸奶器？",
            expected_keywords=["蓝牙", "长按", "切换键", "配对", "SporraMom"],
            expected_section="App 使用指南",
            category="app",
            difficulty="normal",
        ),
        TestCase(
            query="App 连接失败怎么办？",
            expected_keywords=["蓝牙", "未开", "定位权限"],
            expected_section="故障排除",
            category="app",
            difficulty="normal",
        ),

        # === 边界/陷阱问题 ===
        TestCase(
            query="这个吸奶器和美德乐比哪个好？",
            expected_keywords=[],  # 不应编造对比信息
            expected_section="",
            category="out_of_scope",
            difficulty="hard",
        ),
        TestCase(
            query="吸奶器坏了可以自己修吗？",
            expected_keywords=["售后", "保修", "联系"],
            expected_section="保修服务",
            category="edge",
            difficulty="hard",
        ),
    ]

    return test_cases


# ============================================================
# 评测指标计算
# ============================================================

def compute_hit_rate(retrieved_sections: List[str], expected_section: str, k: int) -> float:
    """
    Hit Rate@K: top-K 结果中是否包含正确文档。

    计算方式: 如果 expected_section 的关键词出现在 top-K 任一结果中，得 1 分。
    """
    if not expected_section:
        return 1.0  # 没有期望章节时默认通过

    top_k = retrieved_sections[:k]
    expected_lower = expected_section.lower()

    for section in top_k:
        # 模糊匹配: 期望关键词出现在实际章节路径中
        if expected_lower in section.lower() or any(
            kw in section.lower() for kw in expected_lower.split(" > ")
        ):
            return 1.0
    return 0.0


def compute_mrr(retrieved_sections: List[str], expected_section: str) -> float:
    """
    MRR (Mean Reciprocal Rank): 正确文档排名的倒数。

    如果正确文档排在第 1 位 → MRR = 1.0
    如果正确文档排在第 2 位 → MRR = 0.5
    如果正确文档排在第 3 位 → MRR = 0.33
    如果未命中 → MRR = 0.0
    """
    if not expected_section:
        return 1.0

    expected_lower = expected_section.lower()
    for rank, section in enumerate(retrieved_sections, start=1):
        if expected_lower in section.lower() or any(
            kw in section.lower() for kw in expected_lower.split(" > ")
        ):
            return 1.0 / rank
    return 0.0


def compute_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """
    关键词覆盖率: 答案中包含多少期望关键词。

    用于衡量:
    - Faithfulness: 关键词来自文档而非编造
    - Completeness: 是否覆盖了问题的所有方面
    """
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords)


def compute_answer_relevance_llm(query: str, answer: str) -> float:
    """
    用 LLM 评判答案相关性 (1-5 分)。

    这是更高级的评测方式，适合面试展示。
    """
    from llm_client import get_llm_client

    prompt = f"""请评判以下回答与问题的相关性，打 1-5 分。

评分标准:
- 5分: 完美回答问题
- 4分: 基本回答了问题，有小瑕疵
- 3分: 部分回答了问题
- 2分: 回答与问题关系不大
- 1分: 完全答非所问

问题: {query}
回答: {answer}

只返回数字分数，不要其他内容。"""

    try:
        llm = get_llm_client()
        resp = llm.chat(prompt, temperature=0.1, max_tokens=10)
        score_str = resp.content.strip()
        # 提取数字
        import re
        numbers = re.findall(r'[1-5]', score_str)
        if numbers:
            return int(numbers[0]) / 5.0
        return 0.5
    except Exception:
        return 0.5


# ============================================================
# 评测执行器
# ============================================================

@dataclass
class EvalResult:
    """单条评测结果"""
    query: str
    category: str
    difficulty: str
    expected_section: str
    expected_keywords: List[str]
    # 检索指标
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    mrr: float = 0.0
    # 生成指标
    keyword_coverage: float = 0.0
    relevance_score: float = 0.0
    # 系统指标
    response_time_ms: int = 0
    fallback_triggered: bool = False
    self_check_passed: bool = True
    # 原始数据
    answer: str = ""
    retrieved_sections: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """评测报告"""
    total_cases: int
    # 检索指标 (平均)
    avg_hit_rate_at_1: float
    avg_hit_rate_at_3: float
    avg_hit_rate_at_5: float
    avg_hit_rate_at_10: float
    avg_mrr: float
    # 生成指标 (平均)
    avg_keyword_coverage: float
    avg_relevance_score: float
    # 系统指标
    avg_response_time_ms: int
    fallback_rate: float
    self_check_pass_rate: float
    # 分类别统计
    category_stats: Dict[str, Dict[str, float]]
    # 明细
    details: List[EvalResult]


def run_evaluation(
    retriever: HybridRetriever,
    test_cases: List[TestCase] = None,
    eval_relevance: bool = False,
) -> EvalReport:
    """
    执行完整评测。

    Args:
        retriever: 已构建好的检索器
        test_cases: 测试用例 (默认使用 generate_test_set())
        eval_relevance: 是否用 LLM 评判相关性 (更准确但更慢)
    """
    if test_cases is None:
        test_cases = generate_test_set()

    results: List[EvalResult] = []
    total = len(test_cases)

    print(f"\n{'='*60}")
    print(f"开始评测 | 测试用例数: {total}")
    print(f"{'='*60}\n")

    for i, tc in enumerate(test_cases, 1):
        print(f"[{i}/{total}] {tc.query[:40]}...", end=" ")

        start_time = time.time()

        # 执行 RAG
        rag_resp = run_rag(tc.query, retriever)
        response_time_ms = int((time.time() - start_time) * 1000)

        # 提取检索到的章节路径
        retrieved_sections = [r.get("section_path", "") for r in rag_resp.retrieval_results]

        # 计算指标
        result = EvalResult(
            query=tc.query,
            category=tc.category,
            difficulty=tc.difficulty,
            expected_section=tc.expected_section,
            expected_keywords=tc.expected_keywords,
            hit_rate_at_1=compute_hit_rate(retrieved_sections, tc.expected_section, 1),
            hit_rate_at_3=compute_hit_rate(retrieved_sections, tc.expected_section, 3),
            hit_rate_at_5=compute_hit_rate(retrieved_sections, tc.expected_section, 5),
            hit_rate_at_10=compute_hit_rate(retrieved_sections, tc.expected_section, 10),
            mrr=compute_mrr(retrieved_sections, tc.expected_section),
            keyword_coverage=compute_keyword_coverage(rag_resp.answer, tc.expected_keywords),
            relevance_score=0.0,
            response_time_ms=response_time_ms,
            fallback_triggered=rag_resp.fallback_triggered,
            self_check_passed=rag_resp.self_check_passed,
            answer=rag_resp.answer,
            retrieved_sections=retrieved_sections,
        )

        # LLM 相关性评测 (可选)
        if eval_relevance:
            result.relevance_score = compute_answer_relevance_llm(tc.query, rag_resp.answer)

        results.append(result)

        status = "✓" if result.keyword_coverage > 0.5 else "✗"
        print(f"{status} HR@3={result.hit_rate_at_3:.0f} MRR={result.mrr:.2f} COV={result.keyword_coverage:.2f} [{response_time_ms}ms]")

    # 汇总
    report = _aggregate_results(results, eval_relevance)

    return report


def _aggregate_results(results: List[EvalResult], has_relevance: bool) -> EvalReport:
    """汇总评测结果"""
    # 分类别统计
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_stats = {}
    for cat, cat_results in categories.items():
        category_stats[cat] = {
            "count": len(cat_results),
            "avg_hit_rate_at_3": np.mean([r.hit_rate_at_3 for r in cat_results]),
            "avg_mrr": np.mean([r.mrr for r in cat_results]),
            "avg_keyword_coverage": np.mean([r.keyword_coverage for r in cat_results]),
            "avg_response_time_ms": int(np.mean([r.response_time_ms for r in cat_results])),
        }

    return EvalReport(
        total_cases=len(results),
        avg_hit_rate_at_1=np.mean([r.hit_rate_at_1 for r in results]),
        avg_hit_rate_at_3=np.mean([r.hit_rate_at_3 for r in results]),
        avg_hit_rate_at_5=np.mean([r.hit_rate_at_5 for r in results]),
        avg_hit_rate_at_10=np.mean([r.hit_rate_at_10 for r in results]),
        avg_mrr=np.mean([r.mrr for r in results]),
        avg_keyword_coverage=np.mean([r.keyword_coverage for r in results]),
        avg_relevance_score=np.mean([r.relevance_score for r in results]) if has_relevance else 0.0,
        avg_response_time_ms=int(np.mean([r.response_time_ms for r in results])),
        fallback_rate=sum(1 for r in results if r.fallback_triggered) / len(results),
        self_check_pass_rate=sum(1 for r in results if r.self_check_passed) / len(results),
        category_stats=category_stats,
        details=results,
    )


def format_eval_report(report: EvalReport) -> str:
    """格式化评测报告为可读文本"""
    lines = []
    lines.append("=" * 60)
    lines.append("SporraMom W3 智能客服系统 — 评测报告")
    lines.append("=" * 60)

    lines.append(f"\n📊 总体指标 (共 {report.total_cases} 个测试用例)")
    lines.append("-" * 40)
    lines.append(f"  检索质量:")
    lines.append(f"    Hit Rate@1:  {report.avg_hit_rate_at_1:.1%}")
    lines.append(f"    Hit Rate@3:  {report.avg_hit_rate_at_3:.1%}")
    lines.append(f"    Hit Rate@5:  {report.avg_hit_rate_at_5:.1%}")
    lines.append(f"    Hit Rate@10: {report.avg_hit_rate_at_10:.1%}")
    lines.append(f"    MRR:         {report.avg_mrr:.3f}")
    lines.append(f"  生成质量:")
    lines.append(f"    关键词覆盖率: {report.avg_keyword_coverage:.1%}")
    if report.avg_relevance_score > 0:
        lines.append(f"    LLM相关性:   {report.avg_relevance_score:.3f}")
    lines.append(f"  系统指标:")
    lines.append(f"    平均响应时间: {report.avg_response_time_ms}ms")
    lines.append(f"    降级触发率:   {report.fallback_rate:.1%}")
    lines.append(f"    自检通过率:   {report.self_check_pass_rate:.1%}")

    lines.append(f"\n📂 分类别统计")
    lines.append("-" * 40)
    for cat, stats in report.category_stats.items():
        lines.append(f"  [{cat}] ({stats['count']}条)")
        lines.append(f"    HR@3={stats['avg_hit_rate_at_3']:.1%}  MRR={stats['avg_mrr']:.3f}  COV={stats['avg_keyword_coverage']:.1%}  耗时={stats['avg_response_time_ms']}ms")

    lines.append(f"\n❌ 失败用例分析")
    lines.append("-" * 40)
    failures = [r for r in report.details if r.keyword_coverage < 0.5 or r.hit_rate_at_3 < 1.0]
    if failures:
        for r in failures[:5]:
            lines.append(f"  Q: {r.query}")
            lines.append(f"    期望章节: {r.expected_section}")
            lines.append(f"    实际命中: {r.retrieved_sections[:3]}")
            lines.append(f"    HR@3={r.hit_rate_at_3:.0f} MRR={r.mrr:.2f} 覆盖率={r.keyword_coverage:.2f}")
            lines.append("")
    else:
        lines.append("  全部通过！")

    return "\n".join(lines)


def save_test_set(test_cases: List[TestCase] = None, path: str = None):
    """保存测试集到 JSON"""
    if test_cases is None:
        test_cases = generate_test_set()
    if path is None:
        path = config.TEST_SET_PATH

    data = [asdict(tc) for tc in test_cases]
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Evaluator] 测试集已保存: {path} ({len(data)} 条)")


if __name__ == "__main__":
    # 保存测试集
    save_test_set()
    print("测试集已生成并保存。")
