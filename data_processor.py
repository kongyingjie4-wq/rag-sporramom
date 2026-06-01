"""
SporraMom W3 智能客服系统 — 数据处理模块

数据来源: 人工按逻辑单元切分的 chunks_manual.json
每个 chunk = 一个独立的知识点，包含口语化描述和同义词注释
"""

import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path

import config


@dataclass
class Chunk:
    """一个知识片段"""
    chunk_id: str        # 唯一标识
    content: str         # 文本内容
    topic: str           # 主题标签 (人工标注)
    char_count: int = 0  # 字符数

    def __post_init__(self):
        self.char_count = len(self.content)

    def to_dict(self):
        return asdict(self)


def _make_chunk_id(content: str, topic: str) -> str:
    """基于内容+主题生成唯一ID"""
    raw = f"{topic}||{content}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_chunks(file_path: str = None) -> List[Chunk]:
    """
    加载人工切分的 chunk 数据。

    数据格式 (chunks_manual.json):
    [
      {"id": 1, "topic": "产品概述", "content": "..."},
      {"id": 2, "topic": "安全警告-仅限哺乳期使用", "content": "..."},
      ...
    ]
    """
    file_path = file_path or os.path.join(os.path.dirname(__file__), "chunks_manual.json")
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Chunk 文件不存在: {file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    chunks = []
    for item in data:
        chunk = Chunk(
            chunk_id=_make_chunk_id(item["content"], item["topic"]),
            content=item["content"],
            topic=item["topic"],
        )
        chunks.append(chunk)

    print(f"[DataProcessor] 加载 {len(chunks)} 个 chunks")
    return chunks


def get_chunks_summary(chunks: List[Chunk]) -> dict:
    """生成 chunks 的统计摘要"""
    total_chars = sum(c.char_count for c in chunks)
    return {
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_chars": total_chars // len(chunks) if chunks else 0,
    }


# 兼容旧接口
load_or_build_chunks = load_chunks


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    chunks = load_chunks()
    summary = get_chunks_summary(chunks)
    print(f"\n=== 数据处理摘要 ===")
    print(f"总 chunk 数: {summary['total_chunks']}")
    print(f"总字符数: {summary['total_chars']}")
    print(f"平均 chunk 长度: {summary['avg_chunk_chars']} 字符")
    print(f"\n所有 chunk 预览:")
    for c in chunks:
        print(f"  [{c.chunk_id}] {c.topic}: {c.content[:60]}...")
