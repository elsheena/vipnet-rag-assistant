"""
02_build_index.py
-----------------
Loads chunks from JSON, computes embeddings using a multilingual
sentence-transformer model, and stores them in a ChromaDB collection.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_FILE = Path(__file__).parent / "data" / "chunks.json"
CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "vipnet_docs"

# Multilingual model — works well for Russian technical text
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[dict], model: SentenceTransformer, collection: chromadb.Collection):
    """Embed chunks in batches and upsert into ChromaDB."""
    texts = [c["text"] for c in chunks]
    ids = [str(c["chunk_id"]) for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    print(f"Embedding {len(texts)} chunks with model '{EMBEDDING_MODEL}'...")
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        # multilingual-e5 requires "passage: " prefix for indexing
        batch_prefixed = [f"passage: {t}" for t in batch]
        embs = model.encode(batch_prefixed, normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend(embs.tolist())

    print("Upserting into ChromaDB...")
    # Upsert in batches to avoid memory issues
    for i in tqdm(range(0, len(ids), BATCH_SIZE)):
        collection.upsert(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=texts[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )
    print(f"Index built: {collection.count()} documents in collection '{COLLECTION_NAME}'")


def main():
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Delete existing collection to rebuild fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    build_index(chunks, model, collection)


if __name__ == "__main__":
    main()
