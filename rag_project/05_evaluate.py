"""
05_evaluate.py
--------------
Runs the RAG pipeline against the benchmark dataset and computes metrics:
  - Hit Rate @ K: fraction of questions where the correct chunk is in top-K results
  - MRR (Mean Reciprocal Rank): average reciprocal rank of the correct chunk
"""

import json
import csv
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BENCHMARK_FILE = Path(__file__).parent / "data" / "benchmark.json"
CHUNKS_FILE = Path(__file__).parent / "data" / "chunks.json"
CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"
RESULTS_FILE = Path(__file__).parent / "data" / "eval_results.csv"
COLLECTION_NAME = "vipnet_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32

TOP_K_VALUES = [1, 3, 5, 10]


def get_or_rebuild_collection(client, chunks, embedder):
    """Get existing collection, or rebuild it if HNSW index is corrupt."""
    try:
        collection = client.get_collection(COLLECTION_NAME)
        # Test if it's readable
        count = collection.count()
        print(f"Collection loaded: {count} documents")
        return collection
    except Exception as e:
        print(f"Collection load failed ({e}), rebuilding index...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        texts = [c["text"] for c in chunks]
        ids = [str(c["chunk_id"]) for c in chunks]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

        print(f"Embedding {len(texts)} chunks...")
        all_embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
            batch = [f"passage: {t}" for t in texts[i : i + BATCH_SIZE]]
            embs = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(embs.tolist())

        print("Upserting into ChromaDB...")
        for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Upserting"):
            collection.upsert(
                ids=ids[i : i + BATCH_SIZE],
                embeddings=all_embeddings[i : i + BATCH_SIZE],
                documents=texts[i : i + BATCH_SIZE],
                metadatas=metadatas[i : i + BATCH_SIZE],
            )
        print(f"Index rebuilt: {collection.count()} documents")
        return collection


def evaluate_retrieval(benchmark, collection, embedder, top_k=10):
    results = []
    for item in tqdm(benchmark, desc="Evaluating retrieval"):
        question = item["question"]
        gt_chunk_id = str(item["ground_truth_chunk_id"])

        # Query prefix is crucial for intfloat/multilingual-e5-large
        query_text = f"query: {question}"
        query_emb = embedder.encode(
            query_text, normalize_embeddings=True
        ).tolist()
        res = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents"],
        )
        retrieved_ids = res["ids"][0]

        rank = next(
            (i + 1 for i, rid in enumerate(retrieved_ids) if rid == gt_chunk_id),
            None,
        )

        result = {
            "id": item["id"],
            "question": question,
            "gt_chunk_id": gt_chunk_id,
            "gt_source": item["ground_truth_source"],
            "retrieved_ids": ",".join(retrieved_ids),
            "rank": rank if rank else -1,
            "rr": (1.0 / rank) if rank else 0.0,
        }
        for k in TOP_K_VALUES:
            result[f"hit@{k}"] = 1 if (rank is not None and rank <= k) else 0
        results.append(result)
    return results


def print_metrics(results):
    n = len(results)
    print(f"\n{'='*50}")
    print(f"Evaluation Results (n={n})")
    print(f"{'='*50}")
    for k in TOP_K_VALUES:
        hit_rate = sum(r[f"hit@{k}"] for r in results) / n
        print(f"  Hit Rate @ {k:2d}: {hit_rate:.3f}")
    mrr = sum(r["rr"] for r in results) / n
    print(f"  MRR:          {mrr:.3f}")
    print(f"{'='*50}")


def main():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} benchmark items")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_or_rebuild_collection(client, chunks, embedder)

    results = evaluate_retrieval(benchmark, collection, embedder, top_k=max(TOP_K_VALUES))
    print_metrics(results)

    fieldnames = list(results[0].keys())
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
