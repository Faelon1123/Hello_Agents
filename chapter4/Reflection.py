from typing import List, Dict, Any, Optional
# 假设 llm_client.py 文件已存在，并从中导入 HelloAgentsLLM 类
from llm_client import HelloAgentsLLM

# --- 模块 1: 记忆模块 ---

class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    def __init__(self):
        # 初始化一个空列表来存储所有记录
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。（将记忆轨迹“序列化”成一段文本）
        """
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

# --- 模块 2: Reflection 智能体 ---

# 1. 初始执行提示词
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

# 2. 反思提示词
REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在**算法效率**上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

# 3. 优化提示词
REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}

# 评审员的反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环：反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n{final_code}")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        # 确保能处理生成器可能返回None的情况
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text

if __name__ == '__main__':
    # 1. 初始化LLM客户端 (请确保你的 .env 和 llm_client.py 文件配置正确)
    try:
        llm_client = HelloAgentsLLM()
    except Exception as e:
        print(f"初始化LLM客户端时出错: {e}")
        exit()

    # 2. 初始化 Reflection 智能体，设置最多迭代2轮
    agent = ReflectionAgent(llm_client, max_iterations=2)

    # 3. 定义任务并运行智能体
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)



# >>> 输出发现回答还是会有bug，比如反思包括【不能认为无需改进】，这也会导致迭代终止
# --- 开始处理任务 ---
# 任务: 编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。

# --- 正在进行初始尝试 ---
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# ```python
# def find_primes(n):
#     """找出 1 到 n 之间（包含 n）所有的素数。

#     使用埃拉托斯特尼筛法（Sieve of Eratosthenes）高效筛选素数。

#     Args:
#         n (int): 搜索范围的上界（包含）。

#     Returns:
#         list[int]: 1 到 n 之间所有素数的升序列表；
#                    若 n < 2，则返回空列表。

#     Examples:
#         >>> find_primes(10)
#         [2, 3, 5, 7]
#         >>> find_primes(1)
#         []
#     """
#     if n < 2:
#         return []

#     is_prime = [True] * (n + 1)
#     is_prime[0] = is_prime[1] = False

#     for i in range(2, int(n ** 0.5) + 1):
#         if is_prime[i]:
#             for multiple in range(i * i, n + 1, i):
#                 is_prime[multiple] = False

#     return [num for num, prime in enumerate(is_prime) if prime]


# if __name__ == "__main__":
#     print(find_primes(50))
# ```
# 📝 记忆已更新，新增一条 'execution' 记录。

# --- 第 1/2 轮迭代 ---

# -> 正在进行反思...
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# # 代码审查反馈

# ## 当前算法分析

# 该代码使用**标准埃拉托斯特尼筛法**，时间复杂度为：

# - 筛选阶段：**O(n log log n)**
# - 收集阶段：O(n)
# - 总体：**O(n log log n)**

# 关键优化点已正确实现：从 `i*i` 开始标记、外层循环仅到 √n。

# ## 存在的算法层面改进空间

# ### 1. 线性筛（欧拉筛）— **O(n)**

# 当前埃氏筛中，一个合数会被其**每个素因子**重复标记（如 60 被 2、3、5 各标记一次）。线性筛保证每个合数**仅被其最小素因子标记一次**，将复杂度降至 **O(n)**：

# ```python
# def find_primes_linear(n):
#     if n < 2:
#         return []
#     is_prime = [True] * (n + 1)
#     is_prime[0] = is_prime[1] = False
#     primes = []
#     for i in range(2, n + 1):
#         if is_prime[i]:
#             primes.append(i)
#         for p in primes:
#             if i * p > n:
#                 break
#             is_prime[i * p] = False
#             if i % p == 0:  # 关键：保证只被最小素因子筛掉
#                 break
#     return primes
# ```

# ### 2. 仅筛奇数（Wheel Optimization）

# 将空间和标记次数减少约 **50%**（渐进常数优化）：

# ```python
# def find_primes_optimized(n):
#     if n < 2:
#         return []
#     if n == 2:
#         return [2]
#     # 只对奇数建筛：索引 i 映射到数字 2i+1
#     sieve_size = (n - 1) // 2  # 奇数 3,5,7,...,n
#     is_prime = [True] * (sieve_size + 1)
#     for i in range(1, (int(n ** 0.5) // 2) + 1):
#         if is_prime[i]:
#             p = 2 * i + 1
#             # 标记 p^2 及其奇数倍，步长 2p
#             start = (p * p - 1) // 2
#             for j in range(start, sieve_size + 1, p):
#                 is_prime[j] = False
#     return [2] + [2 * i + 1 for i in range(1, sieve_size + 1) if is_prime[i]]
# ```

# ## 严重性评级

# | 改进方案 | 复杂度变化 | 实际收益 |
# |---------|-----------|---------|
# | 线性筛 | O(n log log n) → **O(n)** | 理论最优，但 Python 中常数较大 |
# | 仅筛奇数 | 渐进不变，常数减半 | 显著减少一半内存和操作 |
# | 切片赋值替代内层循环 | 渐进不变 | **实际大幅提升**（利用 C 层批量操作） |

# ## 结论

# 当前代码在算法选择上已属优良，但**并非算法最优**。线性筛在渐进复杂度上严格更优（O(n)）。此外，内层循环的逐个赋值在 CPython 中是显著瓶颈，应改用切片赋值：

# ```python
# is_prime[i*i : n+1 : i] = [False] * len(range(i*i, n+1, i))
# ```

# **建议优先级**：切片赋值 > 仅筛奇数 > 线性筛。
# 📝 记忆已更新，新增一条 'reflection' 记录。

# -> 正在进行优化...
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# ```python
# import math


# def find_primes(n: int) -> list[int]:
#     """找出 1 到 n 之间（包含 n）所有的素数。

#     使用埃拉托斯特尼筛法的优化版本，主要包含两项改进：
#     1. 仅对奇数建筛（Wheel 优化），内存占用与标记次数减少约一半；
#     2. 内层标记采用切片赋值，利用 CPython 的 C 层批量操作，
#        避免逐元素赋值带来的解释器开销。

#     索引映射规则：筛数组下标 i 对应实际数字 2*i + 1。

#     Args:
#         n (int): 搜索范围的上界（包含）。

#     Returns:
#         list[int]: 1 到 n 之间所有素数的升序列表；
#                    若 n < 2，则返回空列表。

#     Examples:
#         >>> find_primes(10)
#         [2, 3, 5, 7]
#         >>> find_primes(1)
#         []
#     """
#     if n < 2:
#         return []
#     if n == 2:
#         return [2]

#     # 筛数组下标 i (1 <= i <= sieve_size) 对应数字 2*i + 1，
#     # 覆盖 3, 5, 7, ..., n（或 n-1）中的全部奇数。
#     sieve_size = (n - 1) // 2
#     is_prime = [True] * (sieve_size + 1)

#     # 外层循环只需遍历不超过 sqrt(n) 的奇数所对应的下标。
#     for i in range(1, math.isqrt(n) // 2 + 1):
#         if is_prime[i]:
#             p = 2 * i + 1
#             # 从 p*p 开始以步长 2p 标记 p 的奇数倍；
#             # 映射到筛数组：起点 (p*p - 1) // 2，下标步长 p。
#             start = (p * p - 1) // 2
#             is_prime[start::p] = [False] * len(range(start, sieve_size + 1, p))

#     return [2] + [2 * i + 1 for i in range(1, sieve_size + 1) if is_prime[i]]


# if __name__ == "__main__":
#     print(find_primes(50))
# ```
# 📝 记忆已更新，新增一条 'execution' 记录。

# --- 第 2/2 轮迭代 ---

# -> 正在进行反思...
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# ## 代码评审反馈

# ### 1. 复杂度分析

# - **标记阶段**：对每个奇素数 p ≤ √n，内层切片长度约为 `n/(2p)`，总代价为 `Σ_{p≤√n} n/(2p) = Θ(n log log n)`，且已下沉到 C 层（切片赋值）。
# - **重建阶段**：`Θ(n)`，但在解释器层逐元素执行。
# - **空间**：`list[bool]` 每个槽位是一个 8 字节指针，即 **4n 字节** 主存（n=10⁸ 时约 400 MB），外加临时表（见下）。

# **总复杂度：时间 Θ(n log log n)，空间 Θ(n)。**

# ### 2. 算法层面的结论：无需更换算法

# - 该实现已是最优实用路线：埃氏筛 + 奇数 Wheel + C 层批量标记，不存在“用筛法替代试除法”这类量级的改进空间——代码已经是筛法。
# - **线性筛（欧拉筛）** 虽为 O(n)，但需要逐元素 Python 级标记，在 CPython 中实测慢 5–20 倍，是典型的“渐进更优、实际更差”，不采纳。
# - **Atkin 筛** O(n/log log n) 理论更优，但常数大、实现复杂，纯 Python 下同样是倒退。
# - 注意 log log n 在 n=10¹⁸ 时也仅约 3.7，理论差距无实际意义。

# **因此算法本身达标，但实现层存在显著可削减的常数开销。**

# ### 3. 真正的性能瓶颈（实现层，按严重程度排序）

# **瓶颈 ①（最严重）：`[False] * len(...)` 构造临时 `list`**
# 每次标记都分配一个由 Python 对象指针组成的临时表。所有临时表元素总数约 `(n/2)·ln ln √n`，n=10⁸ 时是数百 MB 级的分配与写穿 churn，且主筛本身 8 B/元素的布局严重损害缓存局部性。

# **瓶颈 ②：末尾的纯 Python 列表推导**
# `2 * i + 1` 算术 + 下标访问 + 条件判断，共 n/2 次解释器迭代（每次 ~50–100 ns），大 n 时墙钟时间往往与整个标记阶段相当甚至更长。

# **瓶颈 ③（次要）**：外层边界 `isqrt(n) // 2 + 1` 最多多迭代一个无害的空切片下标，可收紧为 `(isqrt(n) + 1) // 2`，影响可忽略。

# ### 4. 修复方案

# 用 `bytearray` 替代 `list[bool]`（1 B/元素，`bytes(cnt)` 零填充在 C 层 calloc 完成，无临时对象表），重建阶段用 `itertools.compress` 全程留在 C 层：

# ```python
# import math
# from itertools import compress, islice


# def find_primes(n: int) -> list[int]:
#     if n < 2:
#         return []
#     if n == 2:
#         return [2]

#     sieve_size = (n - 1) // 2
#     sieve = bytearray([1]) * (sieve_size + 1)   # 下标 i ↔ 数字 2i+1

#     for i in range(1, (math.isqrt(n) + 1) // 2):
#         if sieve[i]:
#             p = 2 * i + 1
#             start = (p * p - 1) // 2
#             sieve[start::p] = bytes(len(range(start, sieve_size + 1, p)))

#     return [2, *compress(range(3, n + 1, 2), islice(sieve, 1, None))]
# ```

# 说明：`len(range(...))` 是 O(1)，保留它以保证扩展切片赋值长度严格匹配。正确性与原版等价（已核验 n=1/2/3/4/9/10 边界）。

# ### 5. 预期收益与扩展路径

# - 上述改动在纯 CPython 下大 n 场景通常带来 **2–4 倍**提速，峰值内存降至约 **1/8**（n=10⁸：约 50 MB）。
# - 若允许第三方库，NumPy 布尔筛（同一算法）可再快一个数量级；若 n 超出内存，转向**分块筛 + bytearray**（复杂度不变，换取缓存局部性与有界内存）。
# - Wheel-30 可再省约 20–30% 标记量，但索引映射复杂度大增，性价比低，不建议在此基础上加。

# ### 最终结论

# **算法层面：无需改进**——埃氏筛 + 奇数 Wheel 已是纯 Python 下的复杂度与常数最优解，请勿切换到线性筛或 Atkin 筛。**实现层面必须整改**瓶颈 ①②，否则相当于把算法优势浪费在对象分配和解释器循环上。
# 📝 记忆已更新，新增一条 'reflection' 记录。

# ✅ 反思认为代码已无需改进，任务完成。

# --- 任务完成 ---
# 最终生成的代码:
# ```python
# import math


# def find_primes(n: int) -> list[int]:
#     """找出 1 到 n 之间（包含 n）所有的素数。

#     使用埃拉托斯特尼筛法的优化版本，主要包含两项改进：
#     1. 仅对奇数建筛（Wheel 优化），内存占用与标记次数减少约一半；
#     2. 内层标记采用切片赋值，利用 CPython 的 C 层批量操作，
#        避免逐元素赋值带来的解释器开销。

#     索引映射规则：筛数组下标 i 对应实际数字 2*i + 1。

#     Args:
#         n (int): 搜索范围的上界（包含）。

#     Returns:
#         list[int]: 1 到 n 之间所有素数的升序列表；
#                    若 n < 2，则返回空列表。

#     Examples:
#         >>> find_primes(10)
#         [2, 3, 5, 7]
#         >>> find_primes(1)
#         []
#     """
#     if n < 2:
#         return []
#     if n == 2:
#         return [2]

#     # 筛数组下标 i (1 <= i <= sieve_size) 对应数字 2*i + 1，
#     # 覆盖 3, 5, 7, ..., n（或 n-1）中的全部奇数。
#     sieve_size = (n - 1) // 2
#     is_prime = [True] * (sieve_size + 1)

#     # 外层循环只需遍历不超过 sqrt(n) 的奇数所对应的下标。
#     for i in range(1, math.isqrt(n) // 2 + 1):
#         if is_prime[i]:
#             p = 2 * i + 1
#             # 从 p*p 开始以步长 2p 标记 p 的奇数倍；
#             # 映射到筛数组：起点 (p*p - 1) // 2，下标步长 p。
#             start = (p * p - 1) // 2
#             is_prime[start::p] = [False] * len(range(start, sieve_size + 1, p))

#     return [2] + [2 * i + 1 for i in range(1, sieve_size + 1) if is_prime[i]]


# if __name__ == "__main__":
#     print(find_primes(50))
# ```