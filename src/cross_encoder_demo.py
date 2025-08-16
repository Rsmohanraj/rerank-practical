from sentence_transformers import CrossEncoder


def main():
    pairs = [
        ("The weather today is beautiful", "It's raining!"),
        ("The weather today is beautiful", "Today is a sunny day"),
    ]
    model = CrossEncoder('cross-encoder/stsb-TinyBERT-L-4')
    scores = model.predict(pairs)
    print(scores)


if __name__ == "__main__":
    main()


