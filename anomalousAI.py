from ollama import Client
from dotenv import load_dotenv
from os import getenv
import re
import json

load_dotenv("env.env")

sys_prompt = """
You are a Coherence Classifier AI. Your job is to decide if an input sentence is semantically coherent.

When given a sentence, output only a single JSON object with exactly two fields:
- "thought": your internal reasoning in no more than 20 words
- "coherent": 1 if the sentence is semantically coherent, 0 if not

Do not output anything else.

Example:

User:
I touched the idea.

Assistant:
{"thought":"'Touched' implies physical contact, which doesn't apply to an abstract concept.","coherent":0}
"""

client = Client(host=getenv("OLLAMA_HOST"))


def classify(sentence: str) -> bool:
    """
    Classify a sentence as semantically coherent.
    1 if coherent, 0 if not.
    """
    response = client.chat(
        model="llama3:latest",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": sentence},
        ],
    )

    msg = response.message.content
    print(msg)

    result = json.loads(msg)  # type: ignore
    return bool(result["coherent"])
