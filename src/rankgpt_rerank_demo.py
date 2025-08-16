import os
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.llms.openai import OpenAI


def main():
    # 构造一个小索引
    corpus = [
        "Alan Turing proposed the Turing Test.",
        "The weather today is beautiful.",
        "Ada Lovelace is considered the first computer programmer.",
        "The Turing Award is like the Nobel Prize of computing.",
    ]
    index = VectorStoreIndex.from_documents([d for d in corpus])

    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    query_bundle = QueryBundle("Who proposed the Turing Test?")
    retrieved_nodes = retriever.retrieve(query_bundle)

    # RankGPT: 使用 OpenAI 作为 LLM（或兼容 API）
    reranker = RankGPTRerank(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0),
        top_n=3,
        verbose=True,
    )
    ranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
    print([n.get_content() for n in ranked_nodes])


if __name__ == "__main__":
    main()


