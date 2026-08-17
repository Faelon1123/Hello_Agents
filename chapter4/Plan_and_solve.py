import os
import ast # Python 内置的抽象语法树（Abstract Syntax Tree）标准库模块
from llm_client import HelloAgentsLLM
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量，处理文件不存在异常
try:
    load_dotenv()
except FileNotFoundError:
    print("警告：未找到 .env 文件，将使用系统环境变量。")
except Exception as e:
    print(f"警告：加载 .env 文件时出错: {e}")

# --- 1. LLM客户端定义 ---
# 假设你已经有llm_client.py文件，里面定义了HelloAgentsLLM类

# --- 2. 规划器 (Planner) 定义 ---
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

class Planner:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        # 为了生成计划，我们构建一个简单的消息列表
        messages = [{"role": "user", "content": prompt}]
        
        print("--- 正在生成计划 ---") 
        # 使用流式输出来获取完整的计划
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str) # 用于安全地将包含 Python 字面量（Literal）的字符串转换为对应的真实 Python 数据结构
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

# --- 3. 执行器 (Executor) 定义 ---
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Executor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = ""  # 用于存储历史步骤和结果的字符串
        final_answer = ""
        
        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, plan=plan, history=history if history else "无", current_step=step
            )
            messages = [{"role": "user", "content": prompt}]
            
            response_text = self.llm_client.think(messages=messages) or ""
            
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")
            
        return final_answer # 最后一步的响应就是最终答案

# --- 4. 智能体 (Agent) 整合 ---
class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return
        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

# --- 5. 主函数入口 ---
if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()
        agent = PlanAndSolveAgent(llm_client)
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        agent.run(question)
    except ValueError as e:
        print(e)


# >>>
# --- 开始处理问题 ---
# 问题: 一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？
# --- 正在生成计划 ---
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# ```python
# ["确定周一卖出的苹果数量：15个", "计算周二卖出的苹果数量：周一数量乘以2，即15 × 2 = 30个", "计算周三卖出的苹果数量：周二数量减去5，即30 - 5 = 25个", "计算三天总销售量：周一 + 周二 + 周三，即15 + 30 + 25 = 70个", "得出最终答案：三天总共卖出70个苹果"]
# ```
# ✅ 计划已生成:
# ```python
# ["确定周一卖出的苹果数量：15个", "计算周二卖出的苹果数量：周一数量乘以2，即15 × 2 = 30个", "计算周三卖出的苹果数量：周二数量减去5，即30 - 5 = 25个", "计算三天总销售量：周一 + 周二 + 周三，即15 + 30 + 25 = 70个", "得出最终答案：三天总共卖出70个苹果"]
# ```

# --- 正在执行计划 ---

# -> 正在执行步骤 1/5: 确定周一卖出的苹果数量：15个
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# 周一卖出的苹果数量：15个
# ✅ 步骤 1 已完成，结果: 周一卖出的苹果数量：15个

# -> 正在执行步骤 2/5: 计算周二卖出的苹果数量：周一数量乘以2，即15 × 2 = 30个
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# 周二卖出的苹果数量：15 × 2 = 30个
# ✅ 步骤 2 已完成，结果: 周二卖出的苹果数量：15 × 2 = 30个

# -> 正在执行步骤 3/5: 计算周三卖出的苹果数量：周二数量减去5，即30 - 5 = 25个
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# 周三卖出的苹果数量：30 - 5 = 25个
# ✅ 步骤 3 已完成，结果: 周三卖出的苹果数量：30 - 5 = 25个

# -> 正在执行步骤 4/5: 计算三天总销售量：周一 + 周二 + 周三，即15 + 30 + 25 = 70个
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# 三天总销售量：15 + 30 + 25 = 70个
# ✅ 步骤 4 已完成，结果: 三天总销售量：15 + 30 + 25 = 70个

# -> 正在执行步骤 5/5: 得出最终答案：三天总共卖出70个苹果
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# 三天总共卖出70个苹果。
# ✅ 步骤 5 已完成，结果: 三天总共卖出70个苹果。

# --- 任务完成 ---
# 最终答案: 三天总共卖出70个苹果。
