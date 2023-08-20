import curses
from llm_blocks import chat_utils

def main(stdscr):
    # Clear the screen
    stdscr.clear()

    template = """
    Let's brainstorm some ideas for a new software project.
    I really like working with {language} and {framework} and I'm interested in learning more about {concept}.
    What are some ideas given these constraints? Give me 3 really good ones.
    """

    test_chain = chat_utils.GenericChain(template=template, stream=True)
    response_generator = test_chain("python", "openai", "LLMops")

    # Iterating through the response generator object
    sumstr = ""
    for message in response_generator:
        delta = message['choices'][0]['delta']
        sumstr += delta['content'] if 'content' in delta else ""
        stdscr.addstr(0, 0, sumstr)  # Add text starting from the top-left corner
        stdscr.refresh()  # Refresh the screen to show the changes

    # Exit curses application
    curses.endwin()
    # Print final output to the terminal
    print(sumstr)

# Run the curses application
curses.wrapper(main)
