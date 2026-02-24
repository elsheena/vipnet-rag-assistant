import json
from pathlib import Path

DATA_DIR = Path(r"d:\USER\Загрузки\Новая папка (2)\rag_project\data")

def main():
    with open(DATA_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(DATA_DIR / "benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open(DATA_DIR / "eval_results.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunk_map = {str(c["chunk_id"]): c for c in chunks}

    print("--- EVALUATION DEBUG ---")
    for row in benchmark[:5]: # Check first 5 items
        qid = row["id"]
        question = row["question"]
        gt_id = str(row["ground_truth_chunk_id"])
        
        # Find matching row in CSV (skip header)
        csv_row = next((l for l in lines[1:] if l.startswith(f"{qid},")), None)
        if not csv_row:
            continue
            
        parts = csv_row.strip().split('"')
        retrieved_ids = parts[1].split(",") if len(parts) > 1 else []
        
        print(f"\nQ: {question}")
        print(f"GT ID: {gt_id}")
        if gt_id in chunk_map:
            print(f"GT TEXT: {chunk_map[gt_id]['text'][:300]}...")
        else:
            print("GT ID NOT FOUND IN CHUNKS.JSON!")
            
        print(f"Top 1 ID: {retrieved_ids[0] if retrieved_ids else 'N/A'}")
        if retrieved_ids and retrieved_ids[0] in chunk_map:
            print(f"Top 1 TEXT: {chunk_map[retrieved_ids[0]]['text'][:300]}...")

if __name__ == "__main__":
    main()
