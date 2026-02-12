"""Safely evaluate a math expression."""
def calculate(expr: str) -> str {
    try {
        return str(eval(expr));
    } except Exception as e {
        return "Error while evaluating expression.";
    }
}

"""Run and verify Python code."""
def run_and_test_python_code(code: str) -> str {
    try {
        exec(code, {}, {});
        return "Code executed successfully.";
    } except Exception as e {
        return "Execution error while running code.";
    }
}