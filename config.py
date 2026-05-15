from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CuratorConfig:

    # PDF
    pdf_path: Path = Path("input.pdf")

    # Output
    output_jsonl_path: Path = Path(
        "instruction_finetune.jsonl"
    )

    # Ollama
    ollama_url: str = (
        "http://localhost:11434/api/chat"
    )

    model: str = "qwen2.5:7b"

    # Processing
    pages_per_request: int = 1

    workers: int = 1

    # Text limits
    max_text_chars: int = 2500

    min_text_chars: int = 120

    # Dataset
    min_examples_per_page: int = 2

    max_examples_per_page: int = 4

    min_response_words: int = 15