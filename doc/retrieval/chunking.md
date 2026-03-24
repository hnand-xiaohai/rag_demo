## RAG入门-文件分块（chunking）

### 背景

    大模型都有一个上下文窗口（Content Windows），可以理解为短期记忆，大模型的上下文窗口能同时处理的文本是有限的，不可能一次性把一个几百k的文
    本一次性输入给大模型；以此在把文档输入给模型之前，需要先做一个预处理：把长文档切分成一个个更小的、模型能处理的文本块（chunk），这个过程就叫chunking

#### chunking可能存在的问题

不合理的chunking会直接影响后续整个RAG链路的最终效果

1. **上下文丢失**：一个完整的语义被硬生生切开，导致检索到的文本块信息不全
2. **噪音增加**：切分出的文本块包含了太多不相关的信息，干扰了模型的理解
3. **检索效率下降**：文本块的大小不合适，要么太大抓不住重点，要不太小信息量不足

## 切分方法

以langchain演示

### 固定大小分块（Fixed-Size Chunking）
    设定一个固定的块大小（chunk_size），比如100个字符，然后再设定一个重叠大小（chunk_overlap），比如20个字符，来保证块与块之间有一定的上下文连续性

代码演示
```python
# 简单按照chunk长度简单切分
text = "RAG（Retrieval-Augmented Generation）是一种结合了检索和生成技术的自然语言处理模型。它的核心思想是，在生成答案之前，先从一个大规模的知识库中检索出与问题相关的文档片段，然后将这些片段作为上下文信息，引导生成模型（Generator）产生更准确、更丰富的回答。这个框架显著提升了大型语言模型在处理知识密集型任务时的表现。"


from langchain_text_splitters import CharacterTextSplitter

# 初始化分块器
# chunk_size设置为50个字符，重叠部分为10个字符
text_split = CharacterTextSplitter(
    separator = "",
    chunk_size = 50,
    chunk_overlap = 10,
    length_function = lambda x: len(x),
)

chunks = text_split.split_text(text)

for i,chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(chunk)
    print(f"(长度: {len(chunk)})\n")
```

输出：

```
--- Chunk 1 ---
RAG（Retrieval-Augmented Generation）是一种结合了检索和生成技术的自
(长度: 50)

--- Chunk 2 ---
了检索和生成技术的自然语言处理模型。它的核心思想是，在生成答案之前，先从一个大规模的知识库中检索出与
(长度: 50)

--- Chunk 3 ---
模的知识库中检索出与问题相关的文档片段，然后将这些片段作为上下文信息，引导生成模型（Generato
(长度: 50)

--- Chunk 4 ---
型（Generator）产生更准确、更丰富的回答。这个框架显著提升了大型语言模型在处理知识密集型任务
(长度: 50)

--- Chunk 5 ---
在处理知识密集型任务时的表现。
(长度: 15)
```
**固定大小分块问题**：
1. 句子被无情切断：看看Chunk 1的结尾 "自然语言处"，和Chunk 2的开头 "言处理模型"。一个完整的词“自然语言处理模型”被硬生生劈开
2. 语义完整性被破坏：每个Chunk单独看，都可能不是一个完整的、有意义的句子。当后续的检索系统只召回了其中一个Chunk时，模型得到的就是残缺不全的信息，自然很难做出高质量的回答

**结论**：固定大小分块虽然实现简单，但在大多数场景下，它都是一个比较糟糕的选择。因为它完全忽略了文本的语义结构，只是机械地进行切割

### 递归字符分块（Recursive Character Text Splitting）
    它维护一个分隔符列表 (separators) ，默认情况下是 ["\n\n", "\n", " ", ""]。
    \n\n (双换行符)：代表段落。
    \n (换行符)：代表行。
    ``(空格)：代表单词。
    "" (空字符串)：代表字符。

    它会优先尝试用列表中的第一个分隔符（\n\n）进行分割。如果分割后的文本块仍然大于我们设定的 chunk_size，它就会“递归地”进入下一层，
    用第二个分隔符（\n）对这个过大的块进行再分割，以此类推，直到切分出的文本块小于 chunk_size 为止。

代码演示：

```python
# 稍微复杂一点的示例文本，包含段落和换行
text = """RAG（Retrieval-Augmented Generation）是一种结合了检索和生成技术的自然语言处理模型。
它的核心思想是，在生成答案之前，先从一个大规模的知识库中检索出与问题相关的文档片段。

然后将这些片段作为上下文信息，引导生成模型（Generator）产生更准确、更丰富的回答。
这个框架显著提升了大型语言模型在处理知识密集型任务时的表现，是当前构建高级问答系统的热门技术。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=80,
    chunk_overlap=10,
    length_function=len,
)
chunks  = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print(f"(长度: {len(chunk)})\n")
```

输出：
```
--- Chunk 1 ---
RAG（Retrieval-Augmented Generation）是一种结合了检索和生成技术的自然语言处理模型。
(长度: 58)

--- Chunk 2 ---
它的核心思想是，在生成答案之前，先从一个大规模的知识库中检索出与问题相关的文档片段。
(长度: 42)

--- Chunk 3 ---
然后将这些片段作为上下文信息，引导生成模型（Generator）产生更准确、更丰富的回答。
(长度: 45)

--- Chunk 4 ---
这个框架显著提升了大型语言模型在处理知识密集型任务时的表现，是当前构建高级问答系统的热门技术。
(长度: 47)
```
**结论**：
1. 尊重段落结构：分块器首先尝试用 \n\n（段落分隔符）进行分割
2. 句子保持完整：第二个段落比较长，分块器会用下一级分隔符 \n 来切分。可以看到，Chunk 2 和 Chunk 3 都是完整的句子，没有任何一个词被粗暴地拆开
3. 语义连贯性强：每一个 chunk 都是一个或多个语义完整的句子，这为后续的向量嵌入和检索步骤打下了坚实的基础。当检索系统召回这样的 chunk 时，它提供给大模型的是高质量、易于理解的上下文
 
### 针对特定语言的分割器 (Language-Specific Splitters)
    核心思想是：不再基于通用的标点符号，而是基于编程语言本身的语法结构（如类、函数、导入语句等）来进行分割

代码：
详细请看：code/rag_chunking_demo3、code/rag_chunking_demo4
```python
# 语言：python
# 使用 RecursiveCharacterTextSplitter.from_language() 方法
# 传入语言枚举值 Language.PYTHON
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=150, chunk_overlap=0
)

# 使用针对Markdown的分割器
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=100, chunk_overlap=0
)
```
**结论**：
1. 处理代码、Markdown等结构化文本时，不要使用通用的文本分割器。
2. 应该使用针对特定语言的分割器 (Language-Specific Splitters)，它能基于语法和标记（如函数、类、Markdown标题）进行更智能、更符合逻辑的分割
3. 在LangChain中，通过 RecursiveCharacterTextSplitter.from_language() 方法可以轻松实现这一点