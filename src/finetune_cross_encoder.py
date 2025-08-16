import os
import logging
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder


def main():
    # 配置：请将数据路径改为你本地的 jsonl 文件（字段：query/positive/negative）
    data_dir = os.getenv("RERANK_DATA_DIR", "./data")
    train_path = os.path.join(data_dir, "rerank_train.jsonl")
    val_path = os.path.join(data_dir, "rerank_val.jsonl")
    model_path = os.getenv("RERANK_BASE_MODEL", "BAAI/bge-small-en-v1.5")
    train_batch_size = int(os.getenv("RERANK_BATCH_SIZE", "8"))
    num_epochs = int(os.getenv("RERANK_NUM_EPOCHS", "1"))
    output_dir = os.getenv("RERANK_OUTPUT_DIR", f"ft_{os.path.basename(model_path)}")

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    model = CrossEncoder(model_path, num_labels=1, max_length=512)

    def load_jsonl(path):
        df = pd.read_json(path, lines=True)
        return df

    train_df = load_jsonl(train_path)
    val_df = load_jsonl(val_path)

    train_samples = [InputExample(texts=[r["query"], r["positive"], r["negative"]]) for _, r in train_df.iterrows()]
    dev_samples = [InputExample(texts=[r["query"], r["positive"], r["negative"]]) for _, r in val_df.iterrows()]

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    warmup_steps = 100
    logging.info(f"Warmup-steps: {warmup_steps}")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        evaluation_steps=100,
        optimizer_params={"lr": 1e-5},
        warmup_steps=warmup_steps,
        output_path=output_dir,
        use_amp=True,
    )
    model.save(output_dir)
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()


