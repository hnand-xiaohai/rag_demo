# build_vector_store.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
import chromadb

from config import settings


def build_knowledge_base():
    print(f"正在加载嵌入模型: {settings.EMBED_MODEL_PATH} ...")
    embed_model = HuggingFaceEmbedding(
        model_name=settings.EMBED_MODEL_PATH,
        trust_remote_code=True
    )

    print("正在初始化 ChromaDB...")
    # 初始化 Chroma 客户端
    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    print(f"正在读取文档目录: {settings.DOCS_DIR} ...")
    documents = SimpleDirectoryReader(settings.DOCS_DIR).load_data()

    if not documents:
        print("未检测到文档，请检查data目录！")
        return

    print(f"读取到 {len(documents)} 个文档...")

    # 1. 文本分割
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    nodes = text_splitter(documents)
    print(f"分割为 {len(nodes)} 个节点...")

    # 2. 初始化 Docstore 并添加节点
    print("正在初始化 Docstore...")
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # 3. 创建 StorageContext
    # 注意：这里我们初始化一个默认的 IndexStore，它会被自动管理
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )

    print("正在构建索引并写入向量数据库...")
    # 4. 构建索引
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # 5. 【核心修改】持久化整个 StorageContext
    # 这会自动生成 docstore.json 和 index_store.json
    print(f"正在保存存储上下文到: {settings.CHROMA_PERSIST_DIR}")
    storage_context.persist(persist_dir=settings.CHROMA_PERSIST_DIR)

    print("知识库构建完成!")


if __name__ == "__main__":
    build_knowledge_base()