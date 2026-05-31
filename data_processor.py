"""
SporraMom W3 智能客服系统 — 数据处理模块

设计思路（面试要点）:
  1. 按 Markdown 标题层级切分，而非固定 token 切分
     - 原因: 固定 token 切分会割裂语义，比如把"安全警告"切成两半
     - 按章节切分保证每个 chunk 是一个完整的知识点
  2. 每个 chunk 带元数据 (section_path)
     - 用途: 评测时可以追溯答案来源，调试时可以分析检索质量
  3. 超长段落做二次切分，带 overlap
     - 避免信息在切分边界丢失
"""

import re
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path

import config


@dataclass
class Chunk:
    """一个知识片段"""
    chunk_id: str            # 唯一标识 (内容哈希)
    content: str             # 文本内容
    section_path: str        # 章节路径, 如 "安全说明 > 警告 > 第1条"
    char_count: int = 0      # 字符数
    source_page: str = ""    # 来源标识

    def __post_init__(self):
        self.char_count = len(self.content)

    def to_dict(self):
        return asdict(self)


def _make_chunk_id(content: str, section_path: str) -> str:
    """基于内容+路径生成唯一ID"""
    raw = f"{section_path}||{content}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    对超长文本做二次切分。
    优先在句号、换行处切分，保持语义完整性。
    """
    if len(text) <= max_chars:
        return [text]

    # 切分点: 句号、问号、感叹号、换行
    split_pattern = re.compile(r'(?<=[。？！\n])')
    sentences = split_pattern.split(text)

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            # 保留 overlap 作为上下文
            current = current[-overlap:] + sent if overlap else sent
        else:
            current += sent

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


def parse_markdown(file_path: str) -> List[Chunk]:
    """
    解析 Markdown 文件，按标题层级切分为 Chunk 列表。

    处理逻辑:
    1. 逐行读取，识别标题行 (# ## ### ...)
    2. 遇到新标题时，将之前积累的文本打包为 chunk
    3. 对超长 chunk 做二次切分
    4. 表格内容作为整体保留，不切分
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"产品说明书不存在: {file_path}")

    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    chunks: List[Chunk] = []
    section_stack: List[str] = []  # 当前章节路径栈
    current_content: List[str] = []
    heading_pattern = re.compile(r'^(#{1,4})\s+(.+)')

    def _flush_content():
        """将积累的内容打包为 chunk"""
        if not current_content:
            return

        content = "\n".join(current_content).strip()
        if not content or len(content) < 10:  # 过短的跳过
            current_content.clear()
            return

        section_path = " > ".join(section_stack) if section_stack else "根目录"

        # 对超长内容做二次切分
        sub_chunks = _split_long_text(content, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP)
        for i, sub in enumerate(sub_chunks):
            suffix = f" (part{i+1})" if len(sub_chunks) > 1 else ""
            chunk = Chunk(
                chunk_id=_make_chunk_id(sub, section_path + suffix),
                content=sub,
                section_path=section_path + suffix,
            )
            chunks.append(chunk)

        current_content.clear()

    for line in lines:
        heading_match = heading_pattern.match(line)

        if heading_match:
            # 遇到新标题 → 先把之前的内容 flush
            _flush_content()

            level = len(heading_match.group(1))  # # = 1, ## = 2, ### = 3
            title = heading_match.group(2).strip()

            # 更新章节栈: 同级或更高级标题弹出
            while len(section_stack) >= level:
                section_stack.pop()
            section_stack.append(title)
        else:
            current_content.append(line)

    # 最后一段
    _flush_content()

    return chunks


def load_or_build_chunks(file_path: str = None, force_rebuild: bool = False) -> List[Chunk]:
    """
    加载缓存的 chunks，或重新构建。
    缓存基于文件内容的 hash，文件变了自动重建。
    """
    file_path = file_path or config.PRODUCT_MANUAL_PATH
    cache_path = Path(config.INDEX_DIR) / "chunks_cache.json"

    # 计算文件 hash
    content = Path(file_path).read_text(encoding="utf-8")
    file_hash = hashlib.md5(content.encode()).hexdigest()[:8]

    # 尝试加载缓存
    if not force_rebuild and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("file_hash") == file_hash:
                print(f"[DataProcessor] 从缓存加载 {len(cached['chunks'])} 个 chunks")
                return [Chunk(**c) for c in cached["chunks"]]
        except (json.JSONDecodeError, KeyError):
            pass

    # 重新构建
    print(f"[DataProcessor] 解析产品说明书: {file_path}")
    chunks = parse_markdown(file_path)
    print(f"[DataProcessor] 生成 {len(chunks)} 个 chunks")

    # 保存缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "file_hash": file_hash,
        "chunks": [c.to_dict() for c in chunks],
    }
    cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return chunks


def get_chunks_summary(chunks: List[Chunk]) -> dict:
    """生成 chunks 的统计摘要，用于评测报告"""
    total_chars = sum(c.char_count for c in chunks)
    sections = set(c.section_path.split(" > ")[0] for c in chunks)
    return {
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_chars": total_chars // len(chunks) if chunks else 0,
        "top_level_sections": sorted(sections),
    }


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    chunks = load_or_build_chunks()
    summary = get_chunks_summary(chunks)
    print(f"\n=== 数据处理摘要 ===")
    print(f"总 chunk 数: {summary['total_chunks']}")
    print(f"总字符数: {summary['total_chars']}")
    print(f"平均 chunk 长度: {summary['avg_chunk_chars']} 字符")
    print(f"顶级章节: {summary['top_level_sections']}")
    print(f"\n前 3 个 chunk 预览:")
    for c in chunks[:3]:
        print(f"  [{c.section_path}] {c.content[:80]}...")
