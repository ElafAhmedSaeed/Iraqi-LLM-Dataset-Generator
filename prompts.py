from __future__ import annotations

from collections.abc import Sequence


BASE_PROMPT = """
You are creating an Iraqi Arabic instruction tuning dataset.

IMPORTANT:

The Iraqi dialect MUST sound like REAL IRAQI PEOPLE.

DO NOT use Modern Standard Arabic in Iraqi answers.

BAD Iraqi words:

- يجب
- ينبغي
- يتم
- يمكن
- لذلك
- حيث
- تعتبر
- يُستخدم
- ما هي
- لماذا

GOOD Iraqi words:

- لازم
- يكدر
- شلون
- ليش
- شنو
- مو
- هاي
- هسه
- لأن
- عبالك
- يعني

VERY IMPORTANT:

The Iraqi response MUST sound conversational and natural.

Examples:

Formal:
"يجب استخدام البروتوكول"

Iraqi:
"لازم نستخدم البروتوكول"

Formal:
"ما هي وظيفة الشبكة؟"

Iraqi:
"شنو وظيفة الشبكة؟"

Formal:
"يمكن للمستخدم الوصول"

Iraqi:
"يكدر المستخدم يوصل"

STRICT RULES:

- Return STRICT VALID JSON ONLY
- No markdown
- No explanations
- No text outside JSON
- Escape quotes correctly
- Iraqi dialect must be REAL IRAQI DIALECT
- Do not use formal Arabic in Iraqi fields
- Technical words can remain English
- Do not generate broken JSON

FORMAT:

{
  "examples": [
    {
      "instruction": "...",
      "response": "...",
      "iraqi_instruction": "...",
      "iraqi_response": "..."
    }
  ]
}
"""


def build_batch_prompt(
    page_numbers: Sequence[int],
    min_examples_per_page: int,
    max_examples_per_page: int,
    book_name: str,
    min_response_words: int,
    text_content: str,
) -> str:

    page_count = len(page_numbers)

    target_min = (
        page_count * min_examples_per_page
    )

    target_max = (
        page_count * max_examples_per_page
    )

    pages = ", ".join(
        str(p)
        for p in page_numbers
    )

    return (
        f"{BASE_PROMPT}\n\n"
        f"BOOK: {book_name}\n"
        f"PAGES: {pages}\n\n"
        f"Generate between "
        f"{target_min} and {target_max} examples.\n\n"
        f"Minimum answer length: "
        f"{min_response_words} words.\n\n"
        f"CONTENT:\n\n"
        f"{text_content}"
    )