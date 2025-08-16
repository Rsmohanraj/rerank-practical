from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode


def main():
    nodes = [TextNode(text="Alan Turing proposed the Turing Test."), TextNode(text="Ada Lovelace wrote the first algorithm.")]
    index = VectorStoreIndex(nodes)
    reranker = FlagEmbeddingReranker(top_n=1, model="BAAI/bge-reranker-base")
    query_engine = index.as_query_engine(node_postprocessors=[reranker], similarity_top_k=5)
    response = query_engine.query("Who proposed the Turing Test?")
    print(str(response))


if __name__ == "__main__":
    main()


