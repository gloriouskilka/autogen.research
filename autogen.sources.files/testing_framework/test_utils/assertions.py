# =============================================================================
# test_utils/assertions.py
# =============================================================================

# Utility functions for asserting expected behaviors.


def assert_function_call(
    function_calls: List[FunctionCall],
    expected_name: str,
    expected_arguments: dict,
):
    assert function_calls, "No function calls were made."
    function_call = function_calls[-1]
    assert function_call.name == expected_name, f"Expected function name '{expected_name}', got '{function_call.name}'"
    arguments = json.loads(function_call.arguments)
    assert arguments == expected_arguments, f"Expected arguments {expected_arguments}, got {arguments}"
