
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