from mteb import MTEB
from sentence_transformers import SentenceTransformer


def main():
    model_name = "zhengquan"  # 你的（微调）重排模型路径或名称
    model = SentenceTransformer(model_name)
    evaluation = MTEB(tasks=["MIRACLReranking"])  # 指定 Reranking 任务
    results = evaluation.run(model, output_folder=f"results/{model_name}")
    print(results)


if __name__ == "__main__":
    main()


