import time
from typing import Dict, Optional

from hello_agents import Config, HelloAgentsLLM, Message, ReflectionAgent
from hello_agents.core.exceptions import HelloAgentsException


DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
    """,
}


class MyReflectionAgent(ReflectionAgent):
    """支持自定义提示词的反思型 Agent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
        max_api_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        if max_iterations < 0:
            raise ValueError("max_iterations 不能小于 0")
        if max_api_retries < 0:
            raise ValueError("max_api_retries 不能小于 0")
        if retry_delay < 0:
            raise ValueError("retry_delay 不能小于 0")

        # 允许只覆盖某一个提示词，其余提示词继续使用默认值。
        prompts = {**DEFAULT_PROMPTS, **(custom_prompts or {})}
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            max_iterations=max_iterations,
            custom_prompts=prompts,
        )
        self.max_api_retries = max_api_retries
        self.retry_delay = retry_delay

        print(f"✅ {name} 初始化完成，最大迭代次数: {max_iterations}")

    def run(self, input_text: str, **kwargs) -> str:
        """执行任务，并通过多轮反思持续优化回答。"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 每次运行都使用一份新记忆，避免不同任务之间相互污染。
        self.memory = type(self.memory)()

        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- 第 {iteration}/{self.max_iterations} 轮迭代 ---")

            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result,
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            if self._needs_no_improvement(feedback):
                print("\n✅ 反思认为结果已无需改进，任务完成。")
                break

            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback,
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))
        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """调用 LLM，并对超时等短暂性网络错误进行有限重试。"""
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.max_api_retries + 1):
            try:
                return self.llm.invoke(messages, **kwargs) or ""
            except HelloAgentsException as exc:
                if attempt >= self.max_api_retries or not self._is_retryable(exc):
                    raise

                delay = self.retry_delay * (2**attempt)
                print(
                    f"⚠️ LLM 请求暂时失败，{delay:g} 秒后进行"
                    f"第 {attempt + 1}/{self.max_api_retries} 次重试..."
                )
                if delay:
                    time.sleep(delay)

        # 循环的每个分支都会 return 或 raise，此处仅用于满足静态检查。
        raise RuntimeError("不可达的代码")

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """识别超时、限流和短暂性连接错误，不重试鉴权等永久错误。"""
        retryable_markers = (
            "timeout",
            "timed out",
            "readtimeout",
            "connecttimeout",
            "connection error",
            "connection reset",
            "rate limit",
            "429",
            "502",
            "503",
            "504",
        )

        current: Optional[BaseException] = exc
        while current is not None:
            description = f"{type(current).__name__}: {current}".lower()
            if any(marker in description for marker in retryable_markers):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _needs_no_improvement(feedback: str) -> bool:
        """判断评审是否明确表示无需改进，避免被否定句误触发。"""
        normalized = " ".join(feedback.strip().lower().split())
        if not normalized:
            return False

        chinese_negative_phrases = (
            "不能认为无需改进",
            "并非无需改进",
            "不是无需改进",
            "不代表无需改进",
        )
        if any(phrase in normalized for phrase in chinese_negative_phrases):
            return False

        return normalized in {
            "无需改进",
            "无需改进。",
            "no need for improvement",
            "no need for improvement.",
        }
