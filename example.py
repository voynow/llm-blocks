from llm_blocks import chat_utils

template = """
Do you know about the LeetCode problem {LeetCodeProblemName}? If you do, can you explain it to me?
Detail the problem description, constraints, and examples.
Outline the steps you would take to solve the problem on a high level.
Write the code for the solution.
Add a summary of the problem and solution.
"""

test_chain = chat_utils.GenericChain(template=template, stream=True, model_name="gpt-4")
response_generator = test_chain("Two Sum")
chat_utils.stream_to_console(response_generator)
