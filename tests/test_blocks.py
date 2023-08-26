import unittest
from llm_blocks import block_factory


class TestLLMBlocks(unittest.TestCase):
    """
    Disclaimer: This is not amazing unit test practice, rather here we are asserting that
    the blocks are working as expected/without error.
    """
    def setUp(self):
        self.system_message = "Encapsulate the number with parentheses"
        self.template = "The user is asking for the following: {query}\nInstead, return the negative of this."
        self.query = "Return the number 5 and nothing else."
        self.follow_up_query = "Return the same response as above"

    def test_block(self):
        block = block_factory.get('block')
        response = block.execute(self.query)
        self.assertIn("5", response)

    def test_block_sys_message(self):
        block = block_factory.get('block', system_message=self.system_message)
        response = block.execute(self.query)
        self.assertIn("(5)", response)

    def test_block_stream(self):
        block = block_factory.get('block', stream=True)
        response = block.execute(self.query)
        self.assertIn("5", response)

    def test_block_stream_sys_message(self):
        block = block_factory.get('block', stream=True, system_message=self.system_message)
        response = block.execute(self.query)
        self.assertIn("(5)", response)

    def test_template(self):
        block = block_factory.get('template', template=self.template)
        response = block.execute(self.query)
        self.assertIn("-5", response)

    def test_template_sys_message(self):
        block = block_factory.get('template', template=self.template, system_message=self.system_message)
        response = block.execute(self.query)
        self.assertIn("(-5)", response)

    def test_template_stream(self):
        block = block_factory.get('template', template=self.template, stream=True)
        response = block.execute(self.query)
        self.assertIn("-5", response)

    def test_template_stream_sys_message(self):
        block = block_factory.get('template', template=self.template, stream=True, system_message=self.system_message)
        response = block.execute(self.query)
        self.assertIn("(-5)", response)

    def test_chat_block(self):
        block = block_factory.get('chat')
        block.execute(self.query)
        response = block.execute(self.follow_up_query)
        self.assertIn("5", response)

    def test_chat_block_sys_message(self):
        block = block_factory.get('chat', system_message=self.system_message)
        block.execute(self.query)
        response = block.execute(self.follow_up_query)
        self.assertIn("(5)", response)

    def test_chat_block_stream(self):
        block = block_factory.get('chat', stream=True)
        block.execute(self.query)
        response = block.execute(self.follow_up_query)
        self.assertIn("5", response)

    def test_chat_block_stream_sys_message(self):
        block = block_factory.get('chat', stream=True, system_message=self.system_message)
        block.execute(self.query)
        response = block.execute(self.follow_up_query)
        self.assertIn("(5)", response)


if __name__ == "__main__":
    unittest.main()
