from __future__ import annotations


class PromptTemplate:
    """Wrap raw text into a model prompt."""

    def __init__(self, template: str) -> None:
        if "{text}" not in template:
            raise ValueError("PromptTemplate must contain '{text}' placeholder.")
        self._template = template

    def apply(self, text: str, **kwargs: str) -> str:
        return self._template.format(text=text, **kwargs)

    def __repr__(self) -> str:
        return f"PromptTemplate({self._template!r})"

    @classmethod
    def from_dict(cls, d: dict) -> "PromptTemplate":
        return cls(d["template"])


FACTUALITY = PromptTemplate("Is the following statement true or false?\n{text}\nAnswer:")
FAITHFULNESS = PromptTemplate(
    "Is the following text faithful to the source material?\n{text}\nAnswer yes or no:"
)
NEGATION = PromptTemplate(
    "The following statement is {label_name}.\n{text}\nComplete the sentence: The statement is "
)
RAW = PromptTemplate("{text}")

PROMPT_TYPES: dict[str, PromptTemplate] = {
    "factuality": FACTUALITY,
    "faithfulness": FAITHFULNESS,
    "negation": NEGATION,
    "raw": RAW,
}
