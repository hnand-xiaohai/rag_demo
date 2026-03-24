# 递归字符分块器
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

