## Rerank 技术实战（示例项目）

本项目依据 `rerank技术.txt` 复刻与整理，提供从 SentenceTransformers 快速上手，到在 LlamaIndex 中集成 Reranker（BCE 与 FlagEmbedding），以及 RankGPT（LLM 作为重排器）的完整示例；同时给出 CrossEncoder 微调脚本与 MTEB/C_MTEB 评测脚本。

---

### 目录结构

```
rerank技术实战项目/
  ├─ README.md
  ├─ requirements.txt
  ├─ env.example
  └─ src/
      ├─ st_quickstart.py
      ├─ cross_encoder_demo.py
      ├─ bi_encoder_demo.py
      ├─ li_rerank_bce.py
      ├─ li_rerank_flagembedding.py
      ├─ rankgpt_rerank_demo.py
      ├─ finetune_cross_encoder.py
      ├─ mteb_eval_reranking.py
      └─ c_mteb_eval.py
```

---

### 环境准备

- Python 3.10（推荐）
- GPU + CUDA（推荐，用于加速 Bi-Encoder/CrossEncoder 推理与训练）

安装依赖：
```
pip install -r requirements.txt
```

复制环境变量：
```
复制 env.example 为 .env 并按需填写（HUGGINGFACE_HUB_TOKEN / OPENAI_API_KEY / OLLAMA_BASE_URL）
```

---

### 快速开始

- SentenceTransformers 编码：
```
python src/st_quickstart.py
```

- CrossEncoder 评分：
```
python src/cross_encoder_demo.py
```

- Bi-Encoder 语料编码：
```
python src/bi_encoder_demo.py
```

- LlamaIndex + BCE Embedding + BCE Reranker：
```
python src/li_rerank_bce.py
```

- LlamaIndex + FlagEmbeddingReranker：
```
python src/li_rerank_flagembedding.py
```

- RankGPT（LLM 作为重排器，需要 OPENAI_API_KEY 或兼容 API）：
```
python src/rankgpt_rerank_demo.py
```

- CrossEncoder 三元组微调（需要本地 jsonl 数据集，见脚本顶部注释）：
```
python src/finetune_cross_encoder.py
```

- MTEB Reranking 评测：
```
python src/mteb_eval_reranking.py
```

- C_MTEB Reranking 评测：
```
python src/c_mteb_eval.py
```

---

### 要点说明

- Rerank 原理：先召回（Bi-Encoder/向量检索）→ 再利用 CrossEncoder 或 LLM 对候选进行重排，提升相关性。
- BCE 模型组合：`maidalun1020/bce-embedding-base_v1` + `maidalun1020/bce-reranker-base_v1`，具备优秀中英双语能力。
- LlamaIndex 集成：通过 `SentenceTransformerRerank` 或 `FlagEmbeddingReranker` 作为 `node_postprocessors` 注入 `QueryEngine`。
- RankGPT：将 LLM 作为 Reranker，参照论文与 LlamaIndex 的 `RankGPTRerank` 实现；需可用的 OpenAI 兼容 API。
- CrossEncoder 微调：脚本基于 SentenceTransformers 三元组数据格式（jsonl），可自行准备数据集路径。
- 评测：提供 MTEB 与 C_MTEB 两条路径；可对 rerank 模型进行标准化对比。

---

### 参考

- SentenceTransformers：`https://www.sbert.net/`
- BCE Embedding/Reranker：`https://huggingface.co/maidalun1020/bce-embedding-base_v1` `https://huggingface.co/maidalun1020/bce-reranker-base_v1`
- RankGPT：`https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/rankGPT/`
- MTEB：`https://github.com/embeddings-benchmark/mteb`
- C_MTEB（FlagEmbedding）：`https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB`


