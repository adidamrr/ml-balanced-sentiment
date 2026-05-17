import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_MODULES = {
    "lexicon": "tests.test_lexicon_sentiment",
    "ensemble": "tests.test_ensemble_sentiment",
}


def run_test(function):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            function()
    except Exception as error:
        return {"name": function.__name__, "status": "failed", "error": str(error)}
    return {"name": function.__name__, "status": "passed", "error": None}


def run_module(module_name):
    module = importlib.import_module(module_name)
    functions = [
        getattr(module, name)
        for name in sorted(dir(module))
        if name.startswith("test_") and callable(getattr(module, name))
    ]
    tests = [run_test(function) for function in functions]
    passed = sum(test["status"] == "passed" for test in tests)
    failed = sum(test["status"] == "failed" for test in tests)
    return {"passed": passed, "failed": failed, "tests": tests}


def build_report():
    approaches = {
        approach: run_module(module_name)
        for approach, module_name in TEST_MODULES.items()
    }
    total = sum(result["passed"] + result["failed"] for result in approaches.values())
    passed = sum(result["passed"] for result in approaches.values())
    failed = sum(result["failed"] for result in approaches.values())
    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
        },
        "approaches": approaches,
    }


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_results.json")
    report = build_report()
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
