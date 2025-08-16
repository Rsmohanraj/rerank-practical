from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def prepare_data():
    url = "https://baike.baidu.com/item/AIGC?fromModule=lemma_search-box"
    docs = TrafilaturaWebReader().load_data([url])
    return docs


def embedding_data(docs):
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, persist_dir="./chroma_langchain_db")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=50)

    embed_model = HuggingFaceEmbedding(model_name="maidalun1020/bce-embedding-base_v1")
    rerank = SentenceTransformerRerank(model="maidalun1020/bce-reranker-base_v1", top_n=3)

    base_index = VectorStoreIndex.from_documents(
        documents=docs,
        transformations=[node_parser],
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return base_index, embed_model, rerank


def get_llm():
    llm = Ollama(model="qwen2:7b-instruct-q4_0", request_timeout=120.0)
    return llm


def main():
    question = "艾伦•图灵的论文叫什么"
    docs = prepare_data()
    llm = get_llm()
    base_index, embed_model, rerank = embedding_data(docs)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.num_output = 512
    Settings.context_window = 3000

    query_engine = base_index.as_query_engine(similarity_top_k=5, node_postprocessors=[rerank])
    response = query_engine.query(question)
    print(str(response))


if __name__ == "__main__":
    main()


