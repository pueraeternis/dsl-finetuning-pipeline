from openai import OpenAI


class LLMClient:
    """
    Client for interacting with an LLM through an OpenAI-compatible API.
    Supports both single-shot generation and chat history for chaining.
    """

    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        """
        Initialize the client and prepares connection parameters.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        # Initialize OpenAI-compatible client for vLLM or OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """
        Send a list of messages (chat history) to the LLM.
        Useful for Chain-of-Thought flows where context is preserved.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # pyright: ignore[reportArgumentType]
                temperature=temperature,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            return content if content else ""

        except Exception as e:
            print("Error calling LLM at %s: %s", self.base_url, e)
            return ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a request to the LLM and returns the raw text response.
        Wrapper around chat() for single-shot requests.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages)
