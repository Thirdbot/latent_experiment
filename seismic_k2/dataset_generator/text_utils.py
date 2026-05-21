import json


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
    return [], text.strip()


def format_reasoning_answer(reasoning, final_answer):
    if reasoning:
        reasoning_steps = [str(step).strip().rstrip(".") for step in reasoning if str(step).strip()]
        reasoning_text = ". ".join(reasoning_steps)
        if reasoning_text:
            return f"{reasoning_text}. {final_answer}"
    return final_answer
