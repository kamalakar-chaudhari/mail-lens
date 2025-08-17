from llama_index.core.llms import ChatMessage
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI


class LLMClient:
    """LLM client that can work with different providers."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = "gpt-5-mini",
        **kwargs,
    ):
        self.provider = provider.lower()

        self.model_name = model

        # Initialize the appropriate LLM client
        if self.provider == "groq":
            self.model = Groq(model=model, **kwargs)
        elif self.provider == "openai":
            self.model = OpenAI(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def chat(self, messages: list[ChatMessage]) -> str:
        return self.model.chat(messages)

    def chat_content(self, messages: list[ChatMessage], **kwargs) -> str:
        response = self.model.chat(messages, **kwargs)
        return response.message.content.strip()

    def complete(self, prompt: str) -> str:
        response = self.model.complete(prompt)
        return response.text.strip()

    def predict(self, prompt: str, **kwargs):
        return self.model.predict(prompt, **kwargs)


openai_llm_client = LLMClient(provider="openai", model="gpt-5-mini")
# groq_llm_client = LLMClient(provider="groq")

# response = openai_llm_client.chat(
#     [
#         ChatMessage(role="user", content="Hello, how are you?"),
#     ]
# )

# # response = groq_llm_client.chat("Hello, how are you?")
# print(response)
