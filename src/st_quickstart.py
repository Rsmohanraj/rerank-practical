from sentence_transformers import SentenceTransformer


def main():
    sentences = ["Hello World", "Hallo Welt"]
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    print(embeddings)


if __name__ == "__main__":
    main()


