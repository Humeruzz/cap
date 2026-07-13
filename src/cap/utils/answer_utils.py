import re


def extract_answer(text, *, format_type="math"):
    text = text.strip()

    first_question_idx = text.find("Question:")
    if first_question_idx > 0:
        text = text[:first_question_idx]

    if format_type == "math":
        if "\\boxed{" in text:
            match = re.search(r"\\boxed\{([^}]+)\}", text)
            if match:
                return clean_number(match.group(1))

        if "####" in text:
            parts = text.split("####")
            if len(parts) > 1:
                answer_part = parts[1].split()[0] if parts[1].strip() else ""
                return clean_number(answer_part)

        numbers = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
        if numbers:
            last_num = numbers[-1]
            return clean_number(last_num[0] if last_num[0] else last_num[1])

    elif format_type == "mcq":
        match = re.search(r"\b([A-E])\b", text)
        if match:
            return match.group(1)

    return None


def clean_number(text):
    text = text.strip()
    for char in [",", "$", "%", "g", "."]:
        text = text.replace(char, "")
    try:
        return str(int(text))
    except Exception:
        return text


def extract_last_number(text):
    """Last number in `text` as a float, or None (GSM8K-style numeric answers)."""
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not matches:
        return None
    raw = re.sub(r"[,$]", "", matches[-1][0] or matches[-1][1]).strip()
    try:
        return float(raw)
    except ValueError:
        return None


def extract_binary(text):
    """Trailing 0 or 1 in `text` as an int, or None (fact-check / faithfulness)."""
    match = re.search(r"([01])\D*$", text)
    return int(match.group(1)) if match else None
