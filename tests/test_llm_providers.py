from unittest.mock import mock_open, patch, MagicMock


class TestQianfanProvider:
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_image_data")
    @patch("app.llm.qianfan_provider.qianfan")
    def test_analyze_image_returns_structured_dict(self, mock_qianfan, mock_file):
        """验证千帆 Provider 返回结构化字典"""
        mock_chat = MagicMock()
        mock_chat.do.return_value = {
            "result": '{"objects": ["人物", "桌子"], "scene_desc": "餐厅内", "weather": "室内", "location": "餐厅", "action": "吃饭", "emotion": "开心", "clothing": ["白T恤"], "people": [{"role": "主角", "traits": "长发"}]}'
        }
        mock_qianfan.ChatCompletion.return_value = mock_chat

        from app.llm.qianfan_provider import QianfanProvider

        provider = QianfanProvider(access_key="ak", secret_key="sk")
        result = provider.analyze_image("/fake/path.jpg")

        assert "objects" in result
        assert "scene_desc" in result
        assert result["emotion"] == "开心"

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_image_data")
    @patch("app.llm.qianfan_provider.qianfan")
    def test_analyze_image_handles_json_parse_error(self, mock_qianfan, mock_file):
        """验证千帆返回非 JSON 时降级处理"""
        mock_chat = MagicMock()
        mock_chat.do.return_value = {"result": "这不是 JSON"}
        mock_qianfan.ChatCompletion.return_value = mock_chat

        from app.llm.qianfan_provider import QianfanProvider

        provider = QianfanProvider(access_key="ak", secret_key="sk")
        result = provider.analyze_image("/fake/path.jpg")

        assert result["scene_desc"] == "这不是 JSON"


class TestOpenAIProvider:
    @patch("app.llm.openai_provider.OpenAI")
    def test_chat_returns_content(self, mock_openai_cls):
        """验证 OpenAI Provider 返回消息内容"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "这是测试回复"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        from app.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="key", base_url="https://api.com/v1")
        result = provider.chat([
            {"role": "user", "content": "你好"}
        ])

        assert result == "这是测试回复"

    @patch("app.llm.openai_provider.OpenAI")
    def test_chat_json_mode(self, mock_openai_cls):
        """验证 JSON 模式调用"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        from app.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="key", base_url="https://api.com/v1")
        result = provider.chat_json([
            {"role": "user", "content": "输出 JSON"}
        ])

        assert result == {"key": "value"}
