from mteb import MTEB
from C_MTEB import *  # noqa: F401,F403 - 引入自定义任务
from sentence_transformers import SentenceTransformer


def main():
    model_name = "bert-base-uncased"
    model = SentenceTransformer(model_name)
    evaluation = MTEB(tasks=["T2Reranking"])  # 来自 C_MTEB 的自定义任务
    results = evaluation.run(model, output_folder=f"zh_results/{model_name}")
    print(results)


if __name__ == "__main__":
    main()


