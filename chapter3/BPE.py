import re, collections

def get_stats(vocab):
    """统计词元对频率"""
    pairs = collections.defaultdict(int) # 生成一个字典，该字典访问不存在的键时，会为这个键自动生成一个int类型的值，而不是抛出KeyError异常
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """合并词元对"""
    v_out = {}
    bigram = re.escape(' '.join(pair)) # 对字符串中所有可能被视为正则表达式运算符的特殊字符进行转义
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# 准备语料库，每个词末尾加上</w>表示结束，并切分好字符
vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
num_merges = 4 # 设置合并次数

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"第{i+1}次合并: {best} -> {''.join(best)}")
    print(f"新词表（部分）: {list(vocab.keys())}")
    print("-" * 20)


# >>>
# 第1次合并: ('u', 'g') -> ug
# 新词表（部分）: ['h ug </w>', 'p ug </w>', 'p u n </w>', 'b u n </w>']
# --------------------
# 第2次合并: ('ug', '</w>') -> ug</w>
# 新词表（部分）: ['h ug</w>', 'p ug</w>', 'p u n </w>', 'b u n </w>']
# --------------------
# 第3次合并: ('u', 'n') -> un
# 新词表（部分）: ['h ug</w>', 'p ug</w>', 'p un </w>', 'b un </w>']
# --------------------
# 第4次合并: ('un', '</w>') -> un</w>
# 新词表（部分）: ['h ug</w>', 'p ug</w>', 'p un</w>', 'b un</w>']