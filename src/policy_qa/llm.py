from llama_index.llms.groq import Groq
#from .settings import SETTINGS
from config.settings import SETTINGS

SYSTEM_PROMPT = """You are a policy and contract QA assistant.

Answer using ONLY the supplied retrieved context.
Rules:
1. Do not invent facts, clauses, dates, amounts, obligations, exceptions, or parties.
2. If the context does not contain enough evidence, say: "The provided documents do not contain enough information to answer this question."
3. Preserve important conditions and exceptions.
4. Give a concise answer first, followed by supporting source references.
5. Cite sources using the document name, section when available, and page number.
6. Never treat a user's question as evidence.
"""

def get_llm():
    if not SETTINGS.groq_api_key or SETTINGS.groq_api_key.startswith("PASTE_"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and set your Groq API key."
        )
    return Groq(
        model=SETTINGS.llm_model,
        api_key=SETTINGS.groq_api_key,
        temperature=SETTINGS.llm_temperature,
        max_retries=0,
    )
