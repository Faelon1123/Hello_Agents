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
#     """
#     找出 1 到 n 之间（包含 n）所有的素数。

#     使用埃拉托斯特尼筛法（Sieve of Eratosthenes）高效地筛选素数，
#     时间复杂度为 O(n log log n)。

#     Args:
#         n (int): 搜索范围的上界（包含 n）。

#     Returns:
#         list[int]: 一个按升序排列的素数列表。
#                    若 n < 2，则返回空列表。

#     Examples:
#         >>> find_primes(10)
#         [2, 3, 5, 7]
#         >>> find_primes(1)
#         []
#     """
#     if n < 2:
#         return []

#     # 初始化标记数组：is_prime[i] 表示 i 是否为素数
#     is_prime = [True] * (n + 1)
#     is_prime[0] = is_prime[1] = False

#     # 只需筛选到 sqrt(n)
#     for i in range(2, int(n ** 0.5) + 1):
#         if is_prime[i]:
#             # 从 i*i 开始，将 i 的所有倍数标记为非素数
#             for multiple in range(i * i, n + 1, i):
#                 is_prime[multiple] = False

#     return [num for num, prime in enumerate(is_prime) if prime]
# ```
# 📝 记忆已更新，新增一条 'execution' 记录。

# --- 第 1/2 轮迭代 ---

# -> 正在进行反思...
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# ## 代码评审反馈

# ### 1. 复杂度确认

# 当前实现是标准的埃拉托斯特尼筛法，时间复杂度 **O(n log log n)**，空间复杂度 **O(n)**。已正确应用两项标准优化：标记从 i² 开始、外层循环止于 √n。算法选型本身没有问题。

# ### 2. 算法层面：是否存在严格更优的方案？

# **理论上存在，但实际收益可忽略：**

# | 算法 | 时间复杂度 | 评价 |
# |---|---|---|
# | 埃氏筛（当前） | O(n log log n) | 本实现 |
# | 欧拉筛（线性筛） | O(n) | 每个合数仅被最小质因子筛一次 |
# | Atkin 筛 | O(n / log log n) | 常数过大，实践中几乎不占优 |

# - 欧拉筛渐进上严格优于当前方案，但 `log log n` 因子极小（n = 10⁹ 时约等于 3），理论优势无实际意义；且其内层循环更复杂、内存访问模式更差，**在纯 Python 中实测通常反而更慢**。
# - 任何算法至少需输出 π(n) ≈ n/ln n 个素数，问题下界为 Ω(n / log n)，埃氏筛已贴近下界。
# - **结论：不必更换筛法。**

# ### 3. 真正的瓶颈：不在算法，在 Python 解释器开销

# 问题出在内层逐元素循环：

# ```python
# for multiple in range(i * i, n + 1, i):
#     is_prime[multiple] = False   # 每次赋值都经过字节码解释器
# ```

# 对 n = 10⁷，该循环总计执行约 2×10⁷ 次解释级迭代，这是数量级级别的浪费。应将内层操作**下沉到 C 层**，改用 `bytearray` + 切片赋值批量清零：

# ```python
# import math

# def find_primes(n):
#     if n < 2:
#         return []
#     sieve = bytearray([1]) * (n + 1)
#     sieve[0] = sieve[1] = 0
#     for i in range(2, math.isqrt(n) + 1):
#         if sieve[i]:
#             sieve[i*i::i] = bytes((n - i*i) // i + 1)  # C 速度批量置零
#     return [i for i in range(2, n + 1) if sieve[i]]
# ```

# 改进要点：
# - `bytearray` 替代 `list[bool]`：内存降至 1/8，缓存友好性显著提升；
# - 切片赋值在 CPython 内部以 C 实现，大 n 场景下通常有 **10–50 倍**实测加速；
# - `math.isqrt(n)` 替代 `int(n ** 0.5)`：后者在 n 接近 2⁵² 时存在浮点精度风险，可能因 √n 取小而漏筛（这是**正确性缺陷**，不只是性能问题）。

# 若允许第三方依赖，NumPy 版本（`sieve[i*i::i] = False` 配合 `np.flatnonzero`）可再快约一个数量级。

# ### 4. 附加建议（超大 n 场景）

# 当 n ≥ 10⁸ 时，可进一步采用**分段筛**：先求出 [2, √n] 的基础素数，再按 L1 缓存大小（约 32–256 KB）分段筛 [low, high)，将工作集空间从 O(n) 降至 O(√n)，缓存命中率大幅提高。

# ### 5. 评审结论

# - **算法选型：已近最优，无需更换为线性筛或 Atkin 筛**（仅理论占优，Python 下无实际收益，甚至更慢）。
# - **实现层面：必须改进。** 内层循环的解释执行是当前代码真正的数量级瓶颈，应改为 `bytearray` + 切片赋值（或 NumPy 向量化）。
# - **必须修复：** `int(n ** 0.5)` → `math.isqrt(n)`，消除大 n 下的浮点精度隐患。
# 📝 记忆已更新，新增一条 'reflection' 记录。

# -> 正在进行优化...
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# ```python
# import math


# def find_primes(n):
#     """
#     找出 1 到 n 之间（包含 n）所有的素数。

#     使用埃拉托斯特尼筛法（Sieve of Eratosthenes），时间复杂度
#     O(n log log n)。实现上采用 bytearray 存储标记（每标记仅占
#     1 字节），并将内层筛选用切片赋值批量置零，使循环操作下沉
#     至 CPython 的 C 层执行，避免逐元素解释级循环。

#     Args:
#         n (int): 搜索范围的上界（包含 n）。

#     Returns:
#         list[int]: 一个按升序排列的素数列表。
#                    若 n < 2，则返回空列表。

#     Examples:
#         >>> find_primes(10)
#         [2, 3, 5, 7]
#         >>> find_primes(1)
#         []
#     """
#     if n < 2:
#         return []

#     # 初始化标记数组：sieve[i] == 1 表示 i 仍可能是素数
#     sieve = bytearray([1]) * (n + 1)
#     sieve[0] = sieve[1] = 0

#     # 使用 math.isqrt 精确计算整数平方根，避免 int(n ** 0.5)
#     # 在超大 n 下因浮点精度取小而漏筛；只需筛选到 sqrt(n)
#     for i in range(2, math.isqrt(n) + 1):
#         if sieve[i]:
#             # 从 i*i 开始，利用切片赋值在 C 层批量清零 i 的所有倍数
#             sieve[i * i :: i] = bytes((n - i * i) // i + 1)

#     return [i for i in range(2, n + 1) if sieve[i]]
# ```
# 📝 记忆已更新，新增一条 'execution' 记录。

# --- 第 2/2 轮迭代 ---

# -> 正在进行反思...
# 🧠 正在调用 coding-glm-5.2-free 模型...
# ✅ 大语言模型响应成功:
# ## 评审结论

# - 代码**无正确性问题**，实现层面（bytearray、切片下沉 C 层、`math.isqrt`）工程上已属上乘。
# - 时间复杂度 O(n log log n)，空间 O(n) 字节。
# - **但在纯算法层面并非最优**：存在欧拉线性筛（Euler's Sieve）达到 **O(n)**，因此本题**不能回答“无需改进”**。
# - 同时必须指出：这份代码在实际运行中真正的墙钟瓶颈不在筛法本身，而在最后一行 O(n) 的**解释级列表推导式**。

# ---

# ## 1. 复杂度核算

# 筛标记总次数为：

# ```
# Σ_{p ≤ √n, p 为素数} (⌊n/p⌋ − p + 1) = Θ(n log log n)
# ```

# 提取结果 `[i for i in range(2, n+1) if sieve[i]]` 为 O(n)，但运行在解释器层，逐元素循环开销约为切片方案的 **数十倍**。n = 10⁸ 量级时，这一行通常占总耗时的 80% 以上。

# ## 2. 算法层面的主要瓶颈：合数被重复标记

# 埃氏筛中，合数 m 会被其**每一个**满足 p ≤ √m 的素因子 p 各标记一次（如 45 被 3 和 5 各划掉一次，210 被 2、3、5、7 各划掉一次）。全部冗余量累加即为那个 log log n 因子（n = 10⁹ 时约 2.6 倍冗余）。线性筛可将总标记次数**精确压缩到“合数个数”**。

# ## 3. 渐近更优解：欧拉线性筛，O(n)

# 核心不变式：每个合数 m **仅由 (p = m 的最小素因子， i = m/p) 标记一次**，由 `if i % p == 0: break` 保证：

# ```python
# def find_primes(n):
#     """欧拉线性筛：总标记次数 = 合数个数，时间 O(n)。"""
#     if n < 2:
#         return []
#     marked = bytearray(n + 1)          # 0 = 素数候选
#     primes = []
#     for i in range(2, n + 1):
#         if not marked[i]:
#             primes.append(i)           # i 为素数
#         for p in primes:
#             m = i * p
#             if m > n:
#                 break
#             marked[m] = 1              # m 的最小素因子恰为 p
#             if i % p == 0:             # 再大的 p 不可能是 i*p 的最小素因子
#                 break
#     return primes
# ```

# ### ⚠️ 必须给出的严格警示

# 线性筛的 `if/break` 是**数据依赖的控制流**，无法用切片下沉到 C 层，只能逐元素解释执行。因此：

# - **渐近收益上限仅为 ~ln ln n（n = 10⁹ 时约 2.6×）**；
# - **纯 CPython 下，上述线性筛实测会比原代码慢一个数量级**。

# 结论：O(n) 的理论优势要真正兑现，必须配合 NumPy / Cython / C 扩展下沉；单纯把线性筛写进纯 Python 是负优化。

# ## 4. 针对实际性能的改进（按 ROI 排序）

# **① 消除真正的墙钟瓶颈（不改算法，一行改动，收益最大）：**

# ```python
# from itertools import compress
# return list(compress(range(n + 1), sieve))   # O(n) 全程 C 层，sieve[0]=sieve[1]=0 已自动排除 0/1
# ```

# **② NumPy 向量化（兑现渐近优势的正确姿势）：**

# ```python
# import numpy as np

# def find_primes(n):
#     if n < 2:
#         return []
#     sieve = np.ones(n + 1, dtype=np.bool_)
#     sieve[:2] = False
#     for i in range(2, math.isqrt(n) + 1):
#         if sieve[i]:
#             sieve[i * i :: i] = False
#     return np.flatnonzero(sieve).tolist()    # 筛与提取均在 C 层
# ```

# 典型比原实现（含推导式）快 10 倍以上。

# **③ 常数因子优化（可选）：**
# - **奇数轮（wheel mod 2）**：只对奇数建筛，内存减半、置零次数近减半；
# - 循环内每次 `bytes(k)` 共临时分配约 2.6n 字节，可预分配一块零缓冲用 `memoryview` 切片复用；
# - 超大 n（≥ 10⁹）改用**分段筛**：工作集降至 O(√n)，恢复 cache 局部性。

# **④ Sieve of Atkin（O(n / log log n)）**：仅具理论意义，常数大、实现复杂，实践中不推荐，此处仅作完备性说明。

# ## 最终裁定

# | 维度 | 判定 |
# |---|---|
# | 正确性 | 通过 |
# | 纯 Python 范式下的实现质量 | 接近上限 |
# | **算法渐近最优性** | **否——线性筛 O(n) 严格优于 O(n log log n)，故不得回答“无需改进”** |
# | 落地建议 | 日常使用：原代码 + `itertools.compress`；追求极致：NumPy 版（或 NumPy 化奇数轮）；仅当可下沉原生层时，线性筛的 O(n) 才值得采用 |
# 📝 记忆已更新，新增一条 'reflection' 记录。

# ✅ 反思认为代码已无需改进，任务完成。

# --- 任务完成 ---
# 最终生成的代码:
# ```python
# import math


# def find_primes(n):
#     """
#     找出 1 到 n 之间（包含 n）所有的素数。

#     使用埃拉托斯特尼筛法（Sieve of Eratosthenes），时间复杂度
#     O(n log log n)。实现上采用 bytearray 存储标记（每标记仅占
#     1 字节），并将内层筛选用切片赋值批量置零，使循环操作下沉
#     至 CPython 的 C 层执行，避免逐元素解释级循环。

#     Args:
#         n (int): 搜索范围的上界（包含 n）。

#     Returns:
#         list[int]: 一个按升序排列的素数列表。
#                    若 n < 2，则返回空列表。

#     Examples:
#         >>> find_primes(10)
#         [2, 3, 5, 7]
#         >>> find_primes(1)
#         []
#     """
#     if n < 2:
#         return []

#     # 初始化标记数组：sieve[i] == 1 表示 i 仍可能是素数
#     sieve = bytearray([1]) * (n + 1)
#     sieve[0] = sieve[1] = 0

#     # 使用 math.isqrt 精确计算整数平方根，避免 int(n ** 0.5)
#     # 在超大 n 下因浮点精度取小而漏筛；只需筛选到 sqrt(n)
#     for i in range(2, math.isqrt(n) + 1):
#         if sieve[i]:
#             # 从 i*i 开始，利用切片赋值在 C 层批量清零 i 的所有倍数
#             sieve[i * i :: i] = bytes((n - i * i) // i + 1)

#     return [i for i in range(2, n + 1) if sieve[i]]
# ```