"""自定义 Plan-and-Solve Agent。

先让 LLM 把复杂问题拆成计划，再把前面步骤的结果作为上下文，
逐步执行计划。
"""

import ast
import re
import time
from typing import Dict, List, Optional

from hello_agents import Config, HelloAgentsLLM, Message, PlanAndSolveAgent
from hello_agents.core.exceptions import HelloAgentsException


# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
计划的最后一步必须整合前面的结果，直接回答用户的原始问题。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """支持自定义提示词、计划解析和失败重试的 Agent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        max_plan_steps: int = 10,
        max_api_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        if max_plan_steps <= 0:
            raise ValueError("max_plan_steps 必须大于 0")
        if max_api_retries < 0:
            raise ValueError("max_api_retries 不能小于 0")
        if retry_delay < 0:
            raise ValueError("retry_delay 不能小于 0")

        prompts = {
            "planner": DEFAULT_PLANNER_PROMPT,
            "executor": DEFAULT_EXECUTOR_PROMPT,
            **(custom_prompts or {}),
        }
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            custom_prompts=prompts,
        )
        self.prompts = prompts
        self.max_plan_steps = max_plan_steps
        self.max_api_retries = max_api_retries
        self.retry_delay = retry_delay

        print(f"✅ {name} 初始化完成，计划最多 {max_plan_steps} 步")

    def run(self, input_text: str, **kwargs) -> str:
        """生成计划、逐步执行，并返回最后一步的结果。"""
        if not input_text or not input_text.strip():
            raise ValueError("输入问题不能为空")

        question = input_text.strip()
        print(f"\n🤖 {self.name} 开始处理问题: {question}")

        plan = self._create_plan(question, **kwargs)
        if not plan:
            return self._finish(question, "无法生成有效的行动计划，任务终止。")
        if len(plan) > self.max_plan_steps:
            return self._finish(
                question,
                f"生成的计划包含 {len(plan)} 步，超过上限 "
                f"{self.max_plan_steps}，任务终止。",
            )

        final_answer = self._execute_plan(question, plan, **kwargs)
        return self._finish(question, final_answer)

    def _create_plan(self, question: str, **kwargs) -> List[str]:
        """请求 LLM 生成计划并解析为字符串列表。"""
        print("\n--- 正在生成计划 ---")
        prompt = self.prompts["planner"].format(question=question)
        response = self._invoke(prompt, **kwargs)
        print(f"✅ 规划器响应:\n{response}")

        plan = self._parse_plan(response)
        if not plan:
            print("❌ 无法从规划器响应中解析出有效步骤")
        return plan

    def _execute_plan(
        self, question: str, plan: List[str], **kwargs
    ) -> str:
        """依次执行每个步骤，后续步骤可看到之前的结果。"""
        history: List[str] = []
        final_answer = ""
        print("\n--- 正在执行计划 ---")

        for index, step in enumerate(plan, start=1):
            print(f"\n-> 正在执行步骤 {index}/{len(plan)}: {step}")
            prompt = self.prompts["executor"].format(
                question=question,
                plan=self._format_plan(plan),
                history="\n\n".join(history) or "无",
                current_step=step,
            )
            final_answer = self._invoke(prompt, **kwargs)
            history.append(f"步骤 {index}: {step}\n结果: {final_answer}")
            print(f"✅ 步骤 {index} 已完成，结果: {final_answer}")

        return final_answer

    def _invoke(self, prompt: str, **kwargs) -> str:
        """调用 LLM，并对超时等短暂性错误做有限重试。"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_api_retries + 1):
            try:
                response = self.llm.invoke(messages, **kwargs)
                # 同时兼容旧版的 str 和新版的 LLMResponse。
                content = response if isinstance(response, str) else getattr(response, "content", "")
                return content or ""
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

        raise RuntimeError("不可达的代码")

    def _finish(self, question: str, answer: str) -> str:
        """统一记录对话历史并返回结果。"""
        print(f"\n--- 任务完成 ---\n最终答案: {answer}")
        self.add_message(Message(question, "user"))
        self.add_message(Message(answer, "assistant"))
        return answer

    @staticmethod
    def _parse_plan(response: str) -> List[str]:
        """解析代码块或纯 Python 列表，并兼容常见编号列表。"""
        if not response or not response.strip():
            return []

        candidates = re.findall(
            r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE
        )
        candidates.append(response.strip())

        # 有些模型会在列表前后附加说明。
        bracketed = re.search(r"\[[\s\S]*\]", response)
        if bracketed:
            candidates.append(bracketed.group(0))

        for candidate in candidates:
            try:
                value = ast.literal_eval(candidate.strip())
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, list):
                steps = [str(item).strip() for item in value if str(item).strip()]
                if steps:
                    return steps

        numbered_steps = re.findall(
            r"^\s*(?:\d+[.)、]|[-*])\s*(.+?)\s*$", response, re.MULTILINE
        )
        return [step.strip() for step in numbered_steps if step.strip()]

    @staticmethod
    def _format_plan(plan: List[str]) -> str:
        return "\n".join(f"{index}. {step}" for index, step in enumerate(plan, 1))

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
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


# 保留文件初稿中的类名，避免旧代码导入失败。
MyPlanSolveAgent = MyPlanAndSolveAgent
