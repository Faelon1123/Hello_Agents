import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    # 智能体的入口
    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []
        current_step = 0

        while current_step < self.max_steps: # ReAct 范式的主体
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。"); break

            # 3. 解析LLM的输出（精确地提取出Thought和Action）
            thought, action = self._parse_output(response_text)
            if thought: print(f"🤔 思考: {thought}")
            if not action: print("警告：未能解析出有效的Action，流程终止。"); break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: 无效的Action格式，请检查。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            # 调用真实工具
            observation = tool_function(tool_input) if tool_function else f"错误：未找到名为 '{tool_name}' 的工具。"
            
            print(f"👀 观察: {observation}")
            # 此处没把Thought加入历史记录，可能的原因是上下文剪枝策略（1、精简 Token 开销 2、避免思维定势 错误累积）
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
  
            # # 打印历史记录 自我验证 【调试】
            # print("="*40)
            # print("\n".join(self.history))
            # print("="*40)

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """
        解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾 (.*?)表示尽可能少地匹配（非贪婪匹配）  $匹配字符串或行的结尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        解析Action字符串，提取工具名称和输入。例如从 Search[华为最新手机] 中提取出工具名 Search 和工具输入 华为最新手机。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""

if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)


# >>>
# 工具 'Search' 已注册。

# --- 第 1 步 ---
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# Thought: 用户询问华为最新发布的手机及其主要卖点。这是一个关于最新产品动态的问题，我的知识可能不是最新的，需要通过搜索引擎获取最新的信息。

# Action: Search[华为最新手机 2024 2025 发布]
# 🤔 思考: 用户询问华为最新发布的手机及其主要卖点。这是一个关于最新产品动态的问题，我的知识可能不是最新的，需要通过搜索引擎获取最新的信息。
# 🎬 行动: Search[华为最新手机 2024 2025 发布]
# 🔍 正在执行 [SerpApi] 网页搜索: 华为最新手机 2024 2025 发布
# 👀 观察: [1] 华为手机- 华为官网
# 探索并选购华为最新手机，了解Mate 系列、Pura 系列、Pocket 系列、nova 系列、畅享系列及相关配件，体验鸿蒙AI 、影像、通信等功能。

# [2] 华为发布会- 华为官网
# 2025.12.22nova 15 系列及全场景新品发布会华为全新发布nova 15 系列、HUAWEI MatePad 11.5 等产品及配件，升级华为鸿蒙智家1 + 2 + N 解决方案及华为鸿蒙智家单品。 2025 ...

# [3] 2026年华为手机各系列介绍及选购指南（8月份更新）暑期 ...
# 近期更新：已更新6月1日发布的华为nova16系列。已更新4月20日发布的华为Pura90系列。已更新3月23日发布的华为Mate80 Pro Max风驰版 ...

# --- 第 2 步 ---
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# Thought: 根据搜索结果，华为最新的手机应该是2026年6月1日发布的nova 16系列（搜索结果显示该选购指南为8月份更新，nova 16系列是最近发布的产品）。但是目前的信息还不够详细，我需要进一步搜索nova 16系列的主要卖点和特色功能，才能完整回答用户的问题。

# Action: Search[华为nova 16系列 发布 主要卖点 配置 特色功能]
# 🤔 思考: 根据搜索结果，华为最新的手机应该是2026年6月1日发布的nova 16系列（搜索结果显示该选购指南为8月份更新，nova 16系列是最近发布的产品）。但是目前的信息还不够详细，我需要进一步搜索nova 16系列的主要卖点和特色功能，才能完整回答用户的问题。
# 🎬 行动: Search[华为nova 16系列 发布 主要卖点 配置 特色功能]
# 🔍 正在执行 [SerpApi] 网页搜索: 华为nova 16系列 发布 主要卖点 配置 特色功能
# 👀 观察: [1] nova 16 系列及全场景新品发布会- 华为官网
# 华为全新发布nova 16 系列、MatePad Pro Max、FreeClip 2 典藏版耳夹耳机等产品及配件。 nova 16 Ultra HUAWEI WATCH FIT 5 Pro 快速预览

# [2] 华为nova 16系列及全场景新品亮相，市场承压下坚守产品力
# 华为nova 16系列坚持核心配置不降档、体验不打折,全系搭载麒麟9010S旗舰芯片,搭配HarmonyOS 6.1与方舟引擎,保障多任务与游戏场景稳定流畅。续航方面实现 ...

# [3] 华为发布nova 16 系列：2 亿主摄，红枫影像，还有十年的答案
# 系统方面，华为nova 16 Pro 搭载HarmonyOS 6.1 系统，支持能将课堂板书跨设备同步的「全景智记6」、课程表导入和小艺深度解题等非常适合学生用户的功能。

# --- 第 3 步 ---
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# Thought: 根据搜索结果，我已经获得了足够的信息。华为最新的手机是2026年6月1日发布的nova 16系列（其中nova 16 Ultra为顶配版本）。主要卖点包括：全系搭载麒麟9010S旗舰芯片、2亿像素主摄和红枫影像技术、HarmonyOS 6.1系统配合方舟引擎，以及针对学生群体的特色功能。现在可以给出最终答案。

# Action: Finish[华为最新的手机是2026年6月1日发布的**华为nova 16系列**（包括nova 16、nova 16 Pro和nova 16 Ultra等型号）。

# 其主要卖点包括：

# 1. **旗舰级芯片**：全系搭载麒麟9010S旗舰芯片，配合HarmonyOS 6.1系统和方舟引擎，保障多任务处理与游戏场景的稳定流畅。

# 2. **强大影像能力**：配备2亿像素主摄，搭载红枫影像技术，拍照表现出色。

# 3. **持久续航**：续航方面有明显提升（官方强调核心配置不降档、体验不打折）。

# 4. **学生友好功能**：nova 16 Pro支持「全景智记6」（可将课堂板书跨设备同步）、课程表导入、小艺深度解题等功能，非常适合学生用户群体。

# 此外，同期发布的还有MatePad Pro Max平板、FreeClip 2典藏版耳夹耳机等全场景新品。如果预算更高或追求商务旗舰，也可以关注Mate 80 Pro Max风驰版（3月发布）或Pura 90系列（4月发布）。]
# 🤔 思考: 根据搜索结果，我已经获得了足够的信息。华为最新的手机是2026年6月1日发布的nova 16系列（其中nova 16 Ultra为顶配版本）。主要卖点包括：全系搭载麒麟9010S旗舰芯片、2亿像素主摄和红枫影像技术、HarmonyOS 6.1系统配合方舟引擎，以及针对学生群体的特色功能。现在可以给出最终答案。
# 🎉 最终答案: 华为最新的手机是2026年6月1日发布的**华为nova 16系列**（包括nova 16、nova 16 Pro和nova 16 Ultra等型号）。

# 其主要卖点包括：

# 1. **旗舰级芯片**：全系搭载麒麟9010S旗舰芯片，配合HarmonyOS 6.1系统和方舟引擎，保障多任务处理与游戏场景的稳定流畅。

# 2. **强大影像能力**：配备2亿像素主摄，搭载红枫影像技术，拍照表现出色。

# 3. **持久续航**：续航方面有明显提升（官方强调核心配置不降档、体验不打折）。

# 4. **学生友好功能**：nova 16 Pro支持「全景智记6」（可将课堂板书跨设备同步）、课程表导入、小艺深度解题等功能，非常适合学生用户群体。

# 此外，同期发布的还有MatePad Pro Max平板、FreeClip 2典藏版耳夹耳机等全场景新品。如果预算更高或追求商务旗舰，也可以关注Mate 80 Pro Max风驰版（3月发布）或Pura 90系列（4月发布）。