import os
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
from datetime import datetime
from llama3 import chat_with_llama, save_conversation, load_system_prompt, main


class TestLlama3(unittest.TestCase):
    def setUp(self):
        self.test_messages = [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": "Test user message"},
            {"role": "assistant", "content": "Test assistant response"}
        ]
        self.test_prompt_file = "test_system_prompt.txt"
        with open(self.test_prompt_file, "w") as file:
            file.write("Test system prompt")

    def tearDown(self):
        if os.path.exists(self.test_prompt_file):
            os.remove(self.test_prompt_file)

    @patch("llama3.requests.post")
    def test_successful_api_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [
            b"data: {\"response\": \"Test response chunk 1\"}\n",
            b"data: {\"response\": \"Test response chunk 2\"}\n",
            b"[DONE]"
        ]
        mock_post.return_value = mock_response
        response_chunks = list(chat_with_llama(self.test_messages))
        self.assertEqual(response_chunks, [
                         "Test response chunk 1", "Test response chunk 2"])
        mock_post.assert_called_once()

    @patch("llama3.requests.post")
    def test_api_request_error(self, mock_post):
        mock_post.side_effect = Exception("Test API error")
        with self.assertRaises(Exception):
            list(chat_with_llama(self.test_messages))
        mock_post.assert_called_once()

    def test_save_conversation(self):
        filename = "test_conversation.txt"
        save_conversation(self.test_messages, filename)
        self.assertTrue(os.path.exists(filename))
        with open(filename, "r") as file:
            content = file.read()
        expected_content = "System: Test system prompt\nUser: Test user message\nAssistant: Test assistant response\n"
        self.assertEqual(content, expected_content)
        os.remove(filename)

    def test_load_system_prompt(self):
        loaded_prompt = load_system_prompt(self.test_prompt_file)
        self.assertEqual(loaded_prompt, "Test system prompt")

    @patch("argparse.ArgumentParser.parse_args")
    @patch("builtins.input", side_effect=["new", "quit"])
    @patch("llama3.save_conversation")
    @patch("llama3.chat_with_llama")
    def test_user_interaction_new_conversation(self, mock_chat, mock_save, mock_input, mock_parse_args):
        mock_chat.return_value = ["Test assistant response"]
        mock_parse_args.return_value = MagicMock(
            system_prompt=self.test_prompt_file, log_level="INFO")
        with patch("sys.stdout", new=StringIO()) as fake_output:
            main()
        self.assertEqual(mock_save.call_count, 2)
        self.assertIn("Previous conversation saved to", fake_output.getvalue())
        self.assertIn("Starting a new conversation.", fake_output.getvalue())

    @patch("argparse.ArgumentParser.parse_args")
    @patch("builtins.input", return_value="quit")
    @patch("llama3.save_conversation")
    def test_user_interaction_quit(self, mock_save, mock_input, mock_parse_args):
        mock_parse_args.return_value = MagicMock(
            system_prompt=self.test_prompt_file, log_level="INFO")
        with patch("sys.stdout", new=StringIO()) as fake_output:
            main()
        self.assertEqual(mock_save.call_count, 1)
        self.assertIn("Conversation saved to", fake_output.getvalue())

    @patch("argparse.ArgumentParser.parse_args")
    @patch("builtins.input", side_effect=["Test user message", "quit"])
    @patch("llama3.chat_with_llama")
    def test_user_interaction_conversation_flow(self, mock_chat, mock_input, mock_parse_args):
        mock_chat.return_value = ["Test assistant response"]
        mock_parse_args.return_value = MagicMock(
            system_prompt=self.test_prompt_file, log_level="INFO")
        with patch("sys.stdout", new=StringIO()) as fake_output:
            main()
        self.assertIn("Test assistant response", fake_output.getvalue())

    @patch("argparse.ArgumentParser.parse_args")
    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("llama3.save_conversation")
    def test_error_handling_keyboard_interrupt(self, mock_save, mock_input, mock_parse_args):
        mock_parse_args.return_value = MagicMock(
            system_prompt=self.test_prompt_file, log_level="INFO")
        with patch("sys.stdout", new=StringIO()) as fake_output:
            main()
        self.assertEqual(mock_save.call_count, 1)
        self.assertIn("Conversation saved to", fake_output.getvalue())


if __name__ == "__main__":
    unittest.main()
