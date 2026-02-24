"""
04_generate_benchmark.py
------------------------
Generates a synthetic benchmark dataset from the extracted chunks.
Uses a simple heuristic: pick chunks that contain specific keywords
and formulate questions. For a production benchmark, you'd use an LLM
to generate Q&A pairs — see the notebook for the LLM-assisted approach.
"""

import json
import random
from pathlib import Path

CHUNKS_FILE = Path(__file__).parent / "data" / "chunks.json"
BENCHMARK_FILE = Path(__file__).parent / "data" / "benchmark.json"

random.seed(42)

# Keyword → question template pairs (Russian technical domain)
QUESTION_TEMPLATES = [
    ("IP-адрес", "Как настроить IP-адрес на устройстве ViPNet Coordinator HW 5?"),
    ("маршрут", "Как добавить статический маршрут в ViPNet Coordinator HW 5?"),
    ("пароль", "Как изменить пароль администратора в ViPNet Coordinator HW 5?"),
    ("VPN", "Как настроить VPN-туннель в ViPNet Coordinator HW 5?"),
    ("интерфейс", "Как просмотреть состояние сетевых интерфейсов?"),
    ("обновление", "Как обновить программное обеспечение ViPNet Coordinator HW 5?"),
    ("лицензия", "Как активировать лицензию ViPNet Coordinator HW 5?"),
    ("журнал", "Как просмотреть журнал событий ViPNet Coordinator HW 5?"),
    ("резервная копия", "Как создать резервную копию конфигурации?"),
    ("DHCP", "Как настроить DHCP-сервер на ViPNet Coordinator HW 5?"),
    ("брандмауэр", "Как настроить правила брандмауэра?"),
    ("SSH", "Как подключиться к устройству по SSH?"),
    ("сброс", "Как выполнить сброс настроек до заводских?"),
    ("NTP", "Как настроить синхронизацию времени по NTP?"),
    ("DNS", "Как настроить DNS-серверы на ViPNet Coordinator HW 5?"),
    ("трансивер", "Какие трансиверы совместимы с ViPNet Coordinator HW 5?"),
    ("CLI", "Как войти в режим командной строки (CLI)?"),
    ("WEB", "Как получить доступ к веб-интерфейсу управления?"),
    ("версия", "Как узнать текущую версию прошивки устройства?"),
    ("подключение", "Как подключить ViPNet Coordinator HW 5 к сети?"),
]


def find_relevant_chunk(chunks: list[dict], keyword: str) -> dict | None:
    """Find a chunk containing the keyword (case-insensitive)."""
    matches = [c for c in chunks if keyword.lower() in c["text"].lower()]
    return random.choice(matches) if matches else None


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    benchmark = []
    for keyword, question in QUESTION_TEMPLATES:
        chunk = find_relevant_chunk(chunks, keyword)
        if chunk is None:
            print(f"  [SKIP] No chunk found for keyword: '{keyword}'")
            continue

        benchmark.append({
            "id": len(benchmark),
            "question": question,
            "keyword": keyword,
            "ground_truth_chunk_id": chunk["chunk_id"],
            "ground_truth_source": chunk["source"],
            "ground_truth_page": chunk["page"],
            "ground_truth_context": chunk["text"],
            # ground_truth_answer is left empty — to be filled manually or by LLM
            "ground_truth_answer": "",
        })

    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"Benchmark saved: {len(benchmark)} questions → {BENCHMARK_FILE}")


if __name__ == "__main__":
    main()
