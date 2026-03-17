"""Safely evaluate a math expression"""
def calculate(expr: str) -> str {
    import ast;
    try {
        return str(ast.literal_eval(expr));
    } except Exception as e {
        return "Error while evaluating expression.";
    }
}

"""Run and verify Python code."""
def run_and_test_python_code(code: str) -> str {
    # TEST FIXTURE ONLY — This function is for integration testing with MockLLM
    # and should NEVER be used as a template for production tools.
    # Executing arbitrary code is inherently unsafe; restrict to numeric-only
    # evaluation (ast.literal_eval) or sandboxed environments in production.
    try {
        exec(code, {}, {});
        return "Code executed successfully.";
    } except Exception as e {
        return "Execution error while running code.";
    }
}
