from __future__ import annotations

import json
from pathlib import Path

from config import CuratorConfig
from curator import run_curation


def resolve_pdf_path():

    default_pdf = Path(
        "input.pdf"
    )

    if default_pdf.is_file():
        return default_pdf

    candidates = sorted(
        Path(".").glob("*.pdf")
    )

    if len(candidates) == 1:
        return candidates[0]

    return default_pdf


def main():

    config = CuratorConfig(

        pdf_path=
            resolve_pdf_path()
    )

    summary = run_curation(
        config
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()