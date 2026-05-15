from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path

import fitz
import requests
from tqdm import tqdm

from config import CuratorConfig
from prompts import build_batch_prompt


@dataclass(slots=True, frozen=True)
class PageData:

    page_number: int

    text: str


# =========================================
# PDF EXTRACTION
# =========================================

def extract_pdf_text(
    pdf_path: Path,
    config: CuratorConfig,
):

    pages = []

    with fitz.open(pdf_path) as document:

        for page_number in tqdm(
            range(len(document)),
            desc="Extracting PDF Text",
        ):

            page = document[page_number]

            text = page.get_text()

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            if (
                len(text)
                < config.min_text_chars
            ):
                continue

            text = text[
                :config.max_text_chars
            ]

            pages.append(

                PageData(
                    page_number=page_number + 1,
                    text=text,
                )
            )

    return pages


# =========================================
# HELPERS
# =========================================

def normalize_book_name(
    pdf_path: Path
):

    name = unicodedata.normalize(
        "NFKC",
        pdf_path.stem,
    )

    name = re.sub(
        r"[_\-]+",
        " ",
        name,
    )

    return re.sub(
        r"\s+",
        " ",
        name,
    ).strip()


def chunked(items, size):

    return [
        items[i:i + size]
        for i in range(
            0,
            len(items),
            size,
        )
    ]


# =========================================
# OLLAMA
# =========================================

def ask_ollama(
    prompt: str,
    config: CuratorConfig,
):

    response = requests.post(

        config.ollama_url,

        json={

            "model": config.model,

            "messages": [

                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            "stream": False,

            "options": {

                "temperature": 0.2,

                "top_p": 0.8,
            }
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


# =========================================
# JSON CLEANING
# =========================================

def clean_json_text(text):

    text = text.strip()

    text = re.sub(
        r"^```json",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```",
        "",
        text,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    return text.strip()


def extract_json(text):

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        return None

    return match.group(0)


def repair_common_json_errors(
    text: str
):

    text = text.replace(
        "\n",
        " "
    )

    text = text.replace(
        "\r",
        " "
    )

    text = re.sub(
        r",\s*}",
        "}",
        text,
    )

    text = re.sub(
        r",\s*]",
        "]",
        text,
    )

    return text


# =========================================
# IRAQI POST PROCESSING
# =========================================

def convert_to_iraqi_style(text):

    replacements = {

        "يجب": "لازم",

        "ينبغي": "لازم",

        "يمكن": "يكدر",

        "يستطيع": "يكدر",

        "ما هي": "شنو",

        "ما هو": "شنو",

        "لماذا": "ليش",

        "كيف": "شلون",

        "هذه": "هاي",

        "هذا": "هذا",

        "يتم": "يصير",

        "لذلك": "لهذا",

        "حيث": "هنا",

        "تعتبر": "تنعتبر",

        "استخدام": "استخدام",

        "الآن": "هسه",

        "بعد ذلك": "بعدين",

        "ليس": "مو",

        "فقط": "بس",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    return text.strip()


# =========================================
# JSON PARSER
# =========================================

def parse_examples_payload(
    raw_text: str
):

    cleaned = clean_json_text(
        raw_text
    )

    extracted = extract_json(
        cleaned
    )

    if not extracted:
        return []

    extracted = repair_common_json_errors(
        extracted
    )

    try:

        payload = json.loads(
            extracted
        )

    except Exception as error:

        print(
            "\nJSON ERROR:"
        )

        print(error)

        return []

    examples = payload.get(
        "examples",
        []
    )

    if not isinstance(
        examples,
        list
    ):
        return []

    valid_examples = []

    for item in examples:

        if not isinstance(
            item,
            dict
        ):
            continue

        instruction = (
            item.get("instruction")
            or item.get("question")
        )

        response = (
            item.get("response")
            or item.get("answer")
        )

        iraqi_instruction = (
            item.get(
                "iraqi_instruction"
            )
            or item.get(
                "iraqi_question"
            )
        )

        iraqi_response = (
            item.get(
                "iraqi_response"
            )
            or item.get(
                "iraqi_answer"
            )
        )

        if not all([
            instruction,
            response,
            iraqi_instruction,
            iraqi_response,
        ]):
            continue

        iraqi_instruction = (
            str(iraqi_instruction)
            .replace('"', "")
            .strip()
        )

        iraqi_response = (
            str(iraqi_response)
            .replace('"', "")
            .strip()
        )

        iraqi_instruction = (
            convert_to_iraqi_style(
                iraqi_instruction
            )
        )

        iraqi_response = (
            convert_to_iraqi_style(
                iraqi_response
            )
        )

        if (
            "Iraqi Answer"
            in iraqi_response
        ):
            continue

        if (
            "response"
            in iraqi_response
        ):
            continue

        valid_examples.append({

            "instruction":
                str(instruction).strip(),

            "response":
                str(response).strip(),

            "iraqi_instruction":
                iraqi_instruction,

            "iraqi_response":
                iraqi_response,
        })

    return valid_examples


# =========================================
# TRAINING FORMAT
# =========================================

def to_training_record(example):

    return {

        "messages": [

            {
                "role": "system",

                "content":
                    "أنت مساعد عراقي "
                    "وتجاوب باللهجة العراقية."
            },

            {
                "role": "user",

                "content":
                    example[
                        "iraqi_instruction"
                    ]
            },

            {
                "role": "assistant",

                "content":
                    example[
                        "iraqi_response"
                    ]
            },
        ],

        "formal_data": {

            "instruction":
                example[
                    "instruction"
                ],

            "response":
                example[
                    "response"
                ],
        },

        "iraqi_data": {

            "instruction":
                example[
                    "iraqi_instruction"
                ],

            "response":
                example[
                    "iraqi_response"
                ],
        },
    }


# =========================================
# SAVE JSONL
# =========================================

def write_jsonl(
    path,
    records
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(

                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )

            file.write("\n")


# =========================================
# BATCH
# =========================================

def run_batch(
    batch_index,
    batch,
    config,
    book_name,
):

    page_numbers = [
        item.page_number
        for item in batch
    ]

    text_content = "\n\n".join(
        item.text
        for item in batch
    )

    prompt = build_batch_prompt(

        page_numbers=
            page_numbers,

        min_examples_per_page=
            config.min_examples_per_page,

        max_examples_per_page=
            config.max_examples_per_page,

        book_name=
            book_name,

        min_response_words=
            config.min_response_words,

        text_content=
            text_content,
    )

    raw_response = ask_ollama(
        prompt,
        config,
    )

    examples = parse_examples_payload(
        raw_response
    )

    return examples


# =========================================
# MAIN PIPELINE
# =========================================

def run_curation(config):

    pdf_path = (
        config.pdf_path
        .expanduser()
        .resolve()
    )

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: "
            f"{pdf_path}"
        )

    book_name = normalize_book_name(
        pdf_path
    )

    pages = extract_pdf_text(
        pdf_path,
        config,
    )

    batches = chunked(
        pages,
        config.pages_per_request,
    )

    all_examples = []

    with ThreadPoolExecutor(
        max_workers=config.workers
    ) as executor:

        futures = {

            executor.submit(
                run_batch,
                index,
                batch,
                config,
                book_name,
            ): index

            for index, batch in enumerate(
                batches,
                start=1,
            )
        }

        for future in tqdm(

            as_completed(futures),

            total=len(futures),

            desc="Generating Iraqi QA",
        ):

            try:

                result = future.result()

                all_examples.extend(
                    result
                )

            except Exception as error:

                print(
                    "\nBATCH ERROR:"
                )

                print(error)

    records = []

    for item in all_examples:

        try:

            records.append(
                to_training_record(
                    item
                )
            )

        except Exception:

            continue

    write_jsonl(
        config.output_jsonl_path,
        records,
    )

    return {

        "book_name":
            book_name,

        "pages":
            len(pages),

        "examples":
            len(records),

        "output":
            str(
                config.output_jsonl_path
            ),
    }