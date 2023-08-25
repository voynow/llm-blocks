import unittest
from unittest.mock import patch
from llm_blocks import blocks

class TestLLMBlocks(unittest.TestCase):

    def setUp(self):
        self.config = blocks.OpenAIConfig()
        self.message_handler = blocks.MessageHandler()
        self.stream_completion = blocks.StreamCompletion()
        self.batch_completion = blocks.BatchCompletion()
        self.template = "Hello, {name}!"

    def execute_block(self, block_type, completion_strategy, input_data):
        """Helper function to execute a block and return the result."""
        block = block_type(self.config, self.message_handler, completion_strategy)
        return block.execute(input_data)

    @patch("openai.ChatCompletion.create")
    def test_block_stream(self, mock_create):
        """Test Block class with streaming enabled."""
        mock_create.return_value = iter([{"choices": [{"delta": {"content": "Hello"}}]}])
        result = self.execute_block(blocks.Block, self.stream_completion, "Test input")
        self.assertEqual(result, "Hello")

    @patch("openai.ChatCompletion.create")
    def test_block_batch(self, mock_create):
        """Test Block class with streaming disabled."""
        mock_create.return_value = {"choices": [{"delta": {"content": "Hello"}}]}
        result = self.execute_block(blocks.Block, self.batch_completion, "Test input")
        self.assertEqual(result, "Hello")

    @patch("openai.ChatCompletion.create")
    def test_template_block(self, mock_create):
        """Test TemplateBlock class with specific template variables."""
        mock_create.return_value = iter([{"choices": [{"delta": {"content": "Hello, John!"}}]}])
        result = self.execute_block(blocks.TemplateBlock, self.stream_completion, {"name": "John"})
        self.assertEqual(result, "Hello, John!")

    @patch("openai.ChatCompletion.create")
    def test_chat_block(self, mock_create):
        """Test ChatBlock class for chat-based interactions."""
        mock_create.return_value = iter([{"choices": [{"delta": {"content": "Hello"}}]}])
        result = self.execute_block(blocks.ChatBlock, self.stream_completion, "Test input")
        self.assertEqual(result, "Hello")

    @patch("openai.ChatCompletion.create")
    def test_system_message(self, mock_create):
        """Test presence of system message in message handler."""
        mock_create.return_value = iter([{"choices": [{"delta": {"content": "Hello"}}]}])
        message_handler = blocks.MessageHandler(system_message="System Init")
        block = blocks.Block(self.config, message_handler, self.stream_completion)
        block.execute("Test input")
        self.assertIn({"role": "system", "content": "System Init"}, message_handler.messages)

if __name__ == "__main__":
    unittest.main()
