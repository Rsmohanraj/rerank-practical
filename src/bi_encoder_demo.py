from sentence_transformers import SentenceTransformer


def main():
    sentences = ["The weather today is beautiful", "It's raining!"]
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 256
    corpus_embeddings = bi_encoder.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    print(corpus_embeddings.shape)


if __name__ == "__main__":
    main()


