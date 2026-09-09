from engine.agent_executor import parse_tool_calls, execute_python_code, execute_web_search, execute_agent_tool

# Test Python Code Execution
res = execute_python_code("x = 25 * 40\nprint(f'Result: {x}')")
print("Python execution:", res)
assert "Result: 1000" in res

# Test Tool Call Parsing
markup = '<thought>I should run code.</thought>\n<tool_calls>[{"name": "python_interpreter", "arguments": {"code": "print(42)"}}]</tool_calls>'
calls = parse_tool_calls(markup)
print("Parsed calls:", calls)
assert len(calls) == 1
assert calls[0]["name"] == "python_interpreter"
assert calls[0]["arguments"] == {"code": "print(42)"}

# Test Tool Routing
tool_res = execute_agent_tool("python_interpreter", {"code": "print('hello from tool')"})
print("Tool result:", tool_res)
assert "hello from tool" in tool_res

print("ALL AGENT EXECUTOR TESTS PASSED!")
