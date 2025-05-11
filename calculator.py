def evaluate_expression(expression):
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
# Calculator tool logic will go here
