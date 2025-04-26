from ollama import Client
from dotenv import load_dotenv
from os import getenv
import re
import json

load_dotenv("env.env")

# sys_prompt = """
# You are a Coherence Classifier AI. Your job is to decide if an input sentence is coherent (makes sense semantically).

# When given a sentence, do the following:

# 1. **Thought:** Output a single line beginning with "Thought:" containing your internal reasoning in no more than 20 words.
# 2. **JSON result:** On the next line, output exactly one JSON object with a single field:
#    - `"coherent"`: use `1` if the sentence is semantically coherent, or `0` if not.

# Do not output anything else.

# Example interaction:

# **User:**
# I touched the idea.

# **Assistant:**
# Thought: "Touched" implies physical contact, which doesn't apply to an abstract concept.
# {"coherent":0}
# """

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
