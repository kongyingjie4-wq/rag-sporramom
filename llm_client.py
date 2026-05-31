"""
SporraMom W3 智能客服系统 — LLM 调用封装

设计思路（面试要点）:
  1. 统一接口: 封装 OpenAI 兼容 API，方便切换模型
  2. 重试机制: 指数退避重试，处理网络抖动
  3. 超时控制: 避免请求挂起
  4. 用量统计: 记录 token 消耗，用于成本分析
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from openai import OpenAI

import config


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str = ""


class LLMClient:
    """LLM 调用客户端，带重试和统计"""

    def __init__(self):
        self.client = OpenAI(
            base_url=config.API_BASE_URL,
            api_key=config.API_KEY,
        )
        self.model = config.LLM_MODEL
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def chat(
        self,
        user_message: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> LLMResponse:
        """
        调用 LLM 生成回答。

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数 (越低越确定性)
            max_tokens: 最大输出 token 数

        Returns:
            LLMResponse 含生成文本和元信息
        """
        temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        max_tokens = max_tokens or config.LLM_MAX_TOKENS

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        # 指数退避重试
        last_error = None
        for attempt in range(3):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=config.LLM_TIMEOUT,
                )
                latency_ms = int((time.time() - start_time) * 1000)

                content = response.choices[0].message.content or ""
                usage = response.usage

                # 统计
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.call_count += 1

                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    model=self.model,
                )

            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"[LLMClient] 调用失败 (attempt {attempt+1}/3): {e}, 等待 {wait_time}s 重试")
                time.sleep(wait_time)

        raise RuntimeError(f"LLM 调用失败 (已重试3次): {last_error}")

    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计"""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_input_tokens": self.total_input_tokens // max(self.call_count, 1),
            "avg_output_tokens": self.total_output_tokens // max(self.call_count, 1),
        }


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


if __name__ == "__main__":
    client = get_llm_client()
    resp = client.chat(
        "你好，请简单介绍一下你自己",
        system_prompt="你是 SporraMom W3 吸奶器的智能客服助手。",
    )
    print(f"回答: {resp.content}")
    print(f"Token: {resp.input_tokens} in / {resp.output_tokens} out")
    print(f"延迟: {resp.latency_ms}ms")
