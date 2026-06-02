"""Loader for the offline development question set.

From disk, the hand-crafted dev questions we read.
Into schemas.Question objects, each JSONL line we transform.
For offline accuracy measurement before live-game submission, used this is.
"""

import json

from schemas import Question, QuestionType


def load_questions(path: str = "data/dev_questions.jsonl") -> list[Question]:
    """Load all questions from a JSONL file and return a list of Question objects.

    From the given path, every non-blank line we parse.
    Into a Question dataclass, each JSON object we convert.
    When option_ids is absent, an empty dict we supply.
    """
    # The questions, here we collect them.
    questions: list[Question] = []

    # From disk, the file we open.
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            # Blank lines, we skip.
            stripped = line.strip()
            if not stripped:
                continue

            # Into a dict, the JSON line we parse.
            record = json.loads(stripped)

            # The qtype string to a QuestionType enum, we map.
            raw_qtype = record.get("qtype", "mcq")
            try:
                qtype = QuestionType(raw_qtype)
            except ValueError:
                # Unknown qtype strings, to UNKNOWN we default.
                qtype = QuestionType.UNKNOWN

            # option_ids absent means offline dev set; an empty dict we use.
            option_ids: dict[str, int] = record.get("option_ids", {})

            # The Question object, we construct it.
            q = Question(
                qid=record["qid"],
                text=record["text"],
                options=record.get("options", {}),
                option_ids=option_ids,
                qtype=qtype,
                level=record.get("level"),
                topic=record.get("topic"),
                language=record.get("language"),
                gold=record.get("gold"),
            )
            questions.append(q)

    # The fully populated list, we return.
    return questions
