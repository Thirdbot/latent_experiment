import json
import re


def parse_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def split_reasoning_answer(text):
    text = str(text or "").strip()
    parsed = parse_json_object(text)
    if isinstance(parsed, dict):
        reasoning = parsed.get("reasoning", [])
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        if not isinstance(reasoning, list):
            reasoning = []
        reasoning = [str(step).strip() for step in reasoning if str(step).strip()]
        final_answer = parsed.get("final_answer") or parsed.get("answer") or ""
        final_answer = str(final_answer).strip()
        if final_answer:
            return reasoning, final_answer
    final_answer_match = re.search(r"(?is)\bfinal[_ ]answer\s*:\s*(.+)$", text)
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip().strip('"')
        reasoning_text = text[: final_answer_match.start()].strip()
        reasoning = []
        try:
            parsed_reasoning = json.loads(reasoning_text)
        except json.JSONDecodeError:
            parsed_reasoning = None
        if isinstance(parsed_reasoning, list):
            reasoning = [str(step).strip() for step in parsed_reasoning if str(step).strip()]
        elif reasoning_text:
            reasoning = [
                line.strip().strip("-*").strip().strip('"')
                for line in reasoning_text.splitlines()
                if line.strip() and line.strip() not in {"[", "]"}
            ]
        if final_answer:
            return reasoning, final_answer
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        reasoning = [str(step).strip() for step in parsed if str(step).strip()]
        if reasoning:
            return reasoning[:-1], reasoning[-1]
    return [], text.strip()


def format_reasoning_answer(reasoning, final_answer):
    if reasoning:
        reasoning_steps = [str(step).strip().rstrip(".") for step in reasoning if str(step).strip()]
        reasoning_text = ". ".join(reasoning_steps)
        if reasoning_text:
            return f"{reasoning_text}. {final_answer}"
    return final_answer
