"""
ViFinQA LLM Client
──────────────────────────────────────────────────────────────
Wrapper for Qwen2.5-Coder-14B using LlamaIndex VLLM integration.
Optimized for Colab T4 (15GB VRAM).
"""

from typing import Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified interface for Code LLM inference via LlamaIndex."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self._llm = None

        logger.info(
            f"LLMClient initialized: model={model_name}, "
            f"gpu_mem={gpu_memory_utilization}, max_len={max_model_len}"
        )

    def load(self) -> None:
        """Load the LLM model (lazy initialization)."""
        if self._llm is not None:
            return

        try:
            from llama_index.llms.vllm import Vllm

            self._llm = Vllm(
                model=self.model_name,
                trust_remote_code=True,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                vllm_kwargs={
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_model_len": self.max_model_len,
                    "enforce_eager": self.enforce_eager,
                },
            )
            logger.info("vLLM model loaded via LlamaIndex successfully")
        except ImportError as e:
            raise ImportError(
                f"Failed to load vLLM. Original error: {e}. "
                "Ensure llama-index-llms-vllm and vllm are installed correctly."
            )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated text string.
        """
        self.load()

        from llama_index.core.llms import ChatMessage, MessageRole

        messages = []
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        response = self._llm.chat(messages)
        return response.message.content

    def get_llama_index_llm(self):
        """Return the underlying LlamaIndex LLM object for direct use."""
        self.load()
        return self._llm
