from .ollama_client import generate

def verify_answer(user_input, evidence_text, draft_answer, model):
    prompt = f"""
    You are a verification assistant.

    Your task is to check whether the draft answer is fully supported by the provided evidence summary.

    Rules:
    - Return exactly one of these labels on the first line:
    SUPPORTED
    PARTIAL
    UNSUPPORTED

    - On the second line, briefly explain why.
    - If the answer is PARTIAL or UNSUPPORTED, include a revised safer answer on later lines.
    - Do not invent facts
    - Be strict about unsupported claims

    User question:
    {user_input}

    Evidence:
    {evidence_text}

    Draft answer:
    {draft_answer}

    Verification result:
    """

    result = generate(model, prompt).strip()
    return result

def parse_verification_result(result_text):
    lines = [line.strip() for line in result_text.splitlines() if line.strip()]

    if not lines:
        return {
            "status": "SUPPORTED",
            "reason": "Empty verifier output; keeping draft answer.",
            "revised_answer": None
        }
    
    first_line = lines[0].upper()

    if first_line not in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}:
        return {
            "status": "SUPPORTED",
            "reason": "Unexpected verifier format; keeping draft answer.",
            "revised_answer": None,
        }
    
    reason = lines[1] if len(lines) > 1 else ""

    revised_answer = None
    if first_line in {"PARTIAL", "UNSUPPORTED"} and len(lines) > 2:
        revised_answer = "\n".join(lines[2:]).strip()

    return {
        "status": first_line,
        "reason": reason,
        "revised_answer": revised_answer or None,
    }

def apply_verification(user_input, evidence_text, draft_answer, model):
    raw_result = verify_answer(user_input, evidence_text, draft_answer, model)
    parsed = parse_verification_result(raw_result)

    status = parsed["status"]

    if status == "SUPPORTED":
        return {
            "final_answer": draft_answer,
            "verification_status": status,
            "verification_reason": parsed["reason"],
        }
    
    if status in {"PARTIAL", "UNSUPPORTED"} and parsed["revised_answer"]:
        return {
            "final_answer": parsed["revised_answer"],
            "verification_status": status,
            "verification_reason": parsed["reason"],
        }
    
    safer_fallback = (
        f"{draft_answer}\n\n"
        "Note: Some parts of this answer may not be fully supported by the retrieved evidence."
    )

    return {
        "final_answer": safer_fallback,
        "verification_status": status,
        "verification_reason": parsed["reason"]
    }