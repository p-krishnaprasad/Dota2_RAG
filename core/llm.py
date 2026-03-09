import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Generic reusable Groq LLM wrapper.
    Not tied to any specific domain.
    """
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def ask(self, prompt: str) -> str:
        """
        Send a prompt to Groq and return the text response.
        Simple wrapper around langchain's invoke.
        """
        response = self.llm.invoke(prompt)
        # response.content extracts just the text from the response object
        return response.content