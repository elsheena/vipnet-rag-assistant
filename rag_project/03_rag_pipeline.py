import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"
COLLECTION_NAME = "vipnet_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Path to the GGUF model file (user must download separately)
# Recommended: Mistral-7B-Instruct-v0.3-Q4_K_M.gguf  (~4.4 GB)
MODEL_PATH = Path(__file__).parent / "models" / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

TOP_K = 5           # number of retrieved chunks
N_GPU_LAYERS = 35   # offload layers to GPU; adjust based on VRAM


SYSTEM_PROMPT = """Ты — технический ассистент по продукту ViPNet Coordinator HW 5.
Отвечай только на основе предоставленного контекста. Если ответа нет в контексте, скажи об этом.
Отвечай на русском языке, кратко и по существу."""


def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query} [/INST]"
    )


def get_or_rebuild_collection(client, collection_name, chunks_file):
    """Try to load the collection; if HNSW fails, rebuild it."""
    try:
        collection = client.get_collection(collection_name)
        # Check if index is readable
        collection.count()
        return collection
    except Exception as e:
        print(f"Collection load failed ({e}), rebuilding index...")
        script_path = Path(__file__).parent / "02_build_index.py"
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
            return client.get_collection(collection_name)
        except Exception as rebuild_error:
            print(f"FAILED TO REBUILD INDEX: {rebuild_error}")
            raise rebuild_error


class RAGPipeline:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        top_k: int = TOP_K,
        n_gpu_layers: int = N_GPU_LAYERS,
    ):
        self.top_k = top_k

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = get_or_rebuild_collection(client, COLLECTION_NAME, Path(__file__).parent / "data" / "chunks.json")
        print(f"  Collection ready.")

        print(f"Loading LLM from: {model_path}")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("RAG pipeline ready.")

    def retrieve(self, query: str) -> list[dict]:
        """Embed query and retrieve top-k relevant chunks."""
        query_emb = self.embedder.encode(
            f"query: {query}", normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": 1 - dist,  # cosine similarity
            })
        return chunks

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate an answer given the query and retrieved context."""
        texts = [c["text"] for c in context_chunks]
        prompt = build_prompt(query, texts)
        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
            stop=["</s>", "[INST]"],
        )
        return response["choices"][0]["text"].strip()

    def query(self, question: str, verbose: bool = True) -> dict:
        """Full RAG query: retrieve + generate."""
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks)
        if verbose:
            print(f"\nQ: {question}")
            print(f"\nA: {answer}")
            print("\nSources:")
            for c in chunks:
                print(f"  [{c['source']}, p.{c['page']}] score={c['score']:.3f}")
        return {"question": question, "answer": answer, "sources": chunks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViPNet RAG Pipeline")
    parser.add_argument("--query", type=str, help="Question to ask the RAG system")
    args = parser.parse_args()

    rag = RAGPipeline()
    
    if args.query:
        rag.query(args.query)
    else:
        while True:
            q = input("\nВведите вопрос (или 'exit'): ").strip()
            if q.lower() in ("exit", "quit", ""):
                break
            rag.query(q)
