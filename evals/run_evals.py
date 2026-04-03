import json
from pathlib import Path

from app.chatbot import ChatBot

def normalize_text(text: str) -> str:
    return (text or "").lower()

def check_keywords(answer, expected_keywords):
    answer_lower = normalize_text(answer)
    missing = []

    for keyword in expected_keywords:
        if keyword.lower() not in answer_lower:
            missing.append(keyword)

    return missing

def check_forbidden_keywords(answer, forbidden_keywords):
    answer_lower = normalize_text(answer)
    found = []

    for keyword in forbidden_keywords:
        if keyword.lower() in answer_lower:
            found.append(keyword)

    return found

def check_sources(actual_sources, expected_sources):
    missing = []

    actual_set = set(actual_sources or [])
    for source in expected_sources or []:
        if source not in actual_set:
            missing.append(source)

    return missing

def evaluate_case(bot, case):
    result = bot.chat_structured(case["input"])

    failures = []

    if result.get("used_tool") != case.get("expected_used_tool"):
        failures.append(
            f'used_tool expected {case.get("expected_used_tool")} but got {result.get("used_tool")}'
        )

    if result.get("tool_name") != case.get("expected_tool_name"):
        failures.append(
            f'tool_name expected {case.get("expected_tool_name")} but got {result.get("tool_name")}'
        )

    expected_verification_status = case.get("expected_verification_status")
    if result.get("verification_status") != expected_verification_status:
        failures.append(
            f'verification_status expected {expected_verification_status} but got {result.get("verification_status")}'
        )

    missing_sources = check_sources(
        result.get("sources", []),
        case.get("expected_sources", [])
    )
    if missing_sources:
        failures.append(f"missing expected sources: {missing_sources}")

    missing_keywords = check_keywords(
        result.get("answer", ""),
        case.get("expected_keywords", [])
    )
    if missing_keywords:
        failures.append(f"missing expected keywords: {missing_keywords}")

    found_forbidden = check_forbidden_keywords(
        result.get("answer", ""),
        case.get("forbidden_keywords", [])
    )
    if found_forbidden:
        failures.append(f"found forbidden keywords: {found_forbidden}")

    return result, failures

def main():
    test_file = Path("evals/test_cases.json")

    with open(test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    failed = 0

    for case in cases:
        bot = ChatBot(persist_memory=False)

        print(f"\n=== Running: {case['name']} ===")

        result, failures = evaluate_case(bot, case)

        if failures:
            failed += 1
            print("FAIL")
            for failure in failures:
                print(f" - {failure}")

            print("\nAnswer:")
            print(result.get("answer", ""))

            print("\nSources:")
            print(result.get("sources", []))
            
            print("\nTool:")
            print(result.get("tool_name"))
            
            print("\nVerification:")
            print(result.get("verification_status"))
        else:
            passed += 1
            print("PASS")
    
    print("\n====================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("\n====================")

if __name__ == "__main__":
    main()