import json
from pathlib import Path
from datetime import datetime

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
    result = bot.chat_structured(case["input"], debug = True)

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

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    
    try: 
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
    
def build_text_report(run_summary, case_results):
    lines = []
    lines.append("=== Eval Run Summary ===")
    lines.append(f"Timestamp: {run_summary['timestamp']}")
    lines.append(f"Passed: {run_summary['passed']}")
    lines.append(f"Failed: {run_summary['failed']}")
    lines.append(f"Total: {run_summary['total']}")
    lines.append("")

    for case in case_results:
        lines.append(f"=== {case['name']} ===")
        lines.append("PASS" if case["passed"] else "FAIL")

        if case["failures"]:
            for failure in case["failures"]:
                lines.append(f" - {failure}")

        lines.append("")
        lines.append("Input:")
        lines.append(case["input"])
        lines.append("")
        lines.append("Answer:")
        lines.append(case["result"].get("answer", ""))
        lines.append("")
        lines.append("Sources:")
        lines.append(str(case["result"].get("sources", [])))
        lines.append("")
        lines.append("Tool:")
        lines.append(str(case["result"].get("tool_name")))
        lines.append("")
        lines.append("Verification:")
        lines.append(str(case["result"].get("verification_status")))
        lines.append("")
        lines.append("Debug:")
        lines.append(json.dumps(sanitize_for_json(case["result"].get("debug", {})), indent=2))
        lines.append("")
        lines.append("=" * 20)
        lines.append("")
    
    return "\n".join(lines)

def save_eval_reports(run_summary, case_results, output_dir="evals/results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_path / f"eval_report_{timestamp}.json"
    txt_path = output_path / f"eval_report_{timestamp}.txt"

    payload = {
        "summary": run_summary,
        "cases": case_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(payload), f, indent=2, ensure_ascii=False)

    text_report = build_text_report(run_summary, case_results)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)

    return str(json_path), str(txt_path)

def main():
    test_file = Path("evals/test_cases.json")
    bot = ChatBot(persist_memory=False)

    with open(test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    failed = 0
    case_results = []

    for case in cases:

        print(f"\n=== Running: {case['name']} ===")

        result, failures = evaluate_case(bot, case)

        is_pass = len(failures) == 0
        if is_pass:
            print("PASS")
            passed += 1
        else:
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

            print("\nDebug:")
            print(json.dumps(result.get("debug", {}), indent=2))
        
        print("\n====================\n")

        case_results.append({
            "name": case["name"],
            "input": case["input"],
            "passed": is_pass,
            "failures": failures,
            "result": sanitize_for_json(result),
        })

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "failed": failed,
        "total": len(cases)
    }
    
    print("\n====================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("\n====================")

    json_path, txt_path = save_eval_reports(run_summary, case_results)

    print("\nSaved reports:")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")

if __name__ == "__main__":
    main()