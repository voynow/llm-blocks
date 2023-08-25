import unittest
from llm_blocks import block_factory


class TestLLMBlocks(unittest.TestCase):


    def test_block(self):
        block = block_factory.get('block')
        response = block.execute("Inlcude the number '1' in your repsonse")
        self.assertIn("1", response)

    def test_block_sys_message(self):
        block = block_factory.get('block', system_message="Inlcude the number '2' in your repsonse")
        response = block.execute("Test input")
        self.assertIn("2", response)

    def test_block_stream(self):
        block = block_factory.get('block', stream=True)
        response = block.execute("Inlcude the number '3' in your repsonse")
        self.assertIn("3", response)

    def test_block_stream_sys_message(self):
        block = block_factory.get('block', stream=True, system_message="Inlcude the number '4' in your repsonse")
        response = block.execute("Test input")
        self.assertIn("4", response)

    def test_template(self):
        block = block_factory.get('template', template="Running unit tests: {query}")
        response = block.execute("Inlcude the number '5' in your repsonse")
        self.assertIn("5", response)

    def test_template_sys_message(self):
        block = block_factory.get('template', template="Running unit tests: {query}", system_message="Inlcude the number '6' in your repsonse")
        response = block.execute("Test input")
        self.assertIn("6", response)

    def test_template_stream(self):
        block = block_factory.get('template', template="Running unit tests: {query}", stream=True)
        response = block.execute("Inlcude the number '7' in your repsonse")
        self.assertIn("7", response)

    def test_template_stream_sys_message(self):
        block = block_factory.get('template', template="Running unit tests: {query}", stream=True, system_message="Inlcude the number '8' in your repsonse")
        response = block.execute("Test input")
        self.assertIn("8", response)


if __name__ == "__main__":
    unittest.main()
