import os
import json
from typing import Any, Dict, Optional

from flask import Flask, request, jsonify

from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# ----------------------------
# Configuration
# ----------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "thenlper/gte-large")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

# Hugging Face Spaces: put your key in "Settings -> Secrets" as OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "In Hugging Face Spaces, add it in Settings -> Secrets."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Prompt + JSON schema (from notebook)
# ----------------------------
DEV_PROMPT = """
You are a Texas Grade 5 Mathematics tutor for kids, and you also support parents and teachers.
Your tone must be kid-safe, friendly, clear, and encouraging. Keep explanations simple, accurate, and non-judgmental.

CRITICAL OUTPUT RULES:
- Output MUST be valid JSON only (no markdown, no code fences, no extra text).
- Use double quotes for all JSON keys/strings.
- Do not include trailing commas.
- Keep responses safe for kids (no unsafe, hateful, sexual, violent, or scary content).

SCOPE RULE:
- Only answer mathematics (Grade 5 level preferred; you may briefly define advanced terms if asked).
- If the user asks anything not related to mathematics, respond ONLY with:
  {"type":"Refusal","message":"Sorry, I can't answer questions other than mathematics."}

OUTPUT TYPES:
1) Concept explanation:
{
  "type": "Concept",
  "message": "Short kid-friendly explanation..."
}

2) Practice questions (MCQ):
- Create up to 5 questions maximum. If user requests more than 5, generate only 5.
- Each question MUST have exactly 4 answer choices.
- The correct answer MUST be one of the 4 choices.
- Provide the correct answer explicitly using "CorrectOption" (A/B/C/D) and "CorrectAnswer" (exact matching text from Answers).
- Keep math appropriate and compute accurately. Avoid trick questions.

{
  "type": "Questions",
  "message": [
    {
      "Q1": "Question text",
      "Answers": { "A": "Option 1", "B": "Option 2", "C": "Option 3", "D": "Option 4" },
      "CorrectOption": "B",
      "CorrectAnswer": "Option 2"
    }
  ]
}

ACCURACY / ANTI-HALLUCINATION RULES:
- Do the math carefully. Ensure only one correct option unless the user explicitly asks for multiple correct answers.
- If you detect ambiguity (missing numbers/units), ask ONE clarifying question using:
  {"type":"Concept","message":"I need one detail to answer: ..."}
  (Keep it math-only and kid-safe.)

STYLE GUIDELINES:
- Use simple words and short sentences for kids.
- For parents/teachers, add a brief note on how to support learning (1–2 sentences).
- Avoid unrelated topics, brand names, or personal data requests.
""".strip()

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "message"],
    "properties": {
        "type": {"type": "string", "enum": ["Concept", "Questions", "Refusal"]},
        "message": {
            "anyOf": [
                {"type": "string"},
                {
                    "type": "array",
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["Q1", "Answers", "CorrectOption", "CorrectAnswer"],
                        "properties": {
                            "Q1": {"type": "string"},
                            "Answers": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["A", "B", "C", "D"],
                                "properties": {
                                    "A": {"type": "string"},
                                    "B": {"type": "string"},
                                    "C": {"type": "string"},
                                    "D": {"type": "string"},
                                },
                            },
                            "CorrectOption": {"type": "string", "enum": ["A", "B", "C", "D"]},
                            "CorrectAnswer": {"type": "string"},
                        },
                    },
                },
            ]
        },
    },
}

# ----------------------------
# Vector DB + Retriever (loaded once at startup)
# ----------------------------
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# ----------------------------
# RAG helpers
# ----------------------------
def generate_context_from_input(user_input: str) -> str:
    """Retrieve relevant chunks from the vector store and return as a single context string."""
    rel_docs = retriever.invoke(user_input)
    context_list = [d.page_content for d in rel_docs]
    return ". ".join(context_list)

def get_llm_response(user_input: str, context: str = "") -> str:
    """Call OpenAI Chat/Responses API and return the model output text (JSON string)."""
    messages = [{"role": "developer", "content": DEV_PROMPT}]

    if context and context.strip():
        messages.append(
            {
                "role": "developer",
                "content": (
                    "Use the following CONTEXT only if it is relevant to the user's request. "
                    "Do not invent facts that are not in the context.\n"
                    "BEGIN_CONTEXT\n"
                    f"{context}\n"
                    "END_CONTEXT"
                ),
            }
        )

    messages.append({"role": "user", "content": user_input})

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
        temperature=0.2,
        max_output_tokens=800,
        text={
            "format": {
                "type": "json_schema",
                "name": "grade5_math_response",
                "strict": True,
                "schema": JSON_SCHEMA,
            }
        },
    )
    return resp.output_text

def generate_response_from_rag(user_input: str) -> Dict[str, Any]:
    context = generate_context_from_input(user_input)
    raw = get_llm_response(user_input=user_input, context=context)

    # Ensure we return valid JSON to the client even if model output is slightly off.
    try:
        return json.loads(raw)
    except Exception:
        return {"type": "Concept", "message": raw}

# ----------------------------
# Flask API
# ----------------------------
app = Flask(__name__)

@app.get("/")
def health():
    return jsonify({"status": "ok"})

@app.post("/MathQuestion")
def math_question():
    payload = request.get_json(silent=True) or {}
    query = payload.get("Query") or payload.get("query")

    if not query or not isinstance(query, str):
        return jsonify({"error": 'Missing required field "Query" (string).'}), 400

    try:
        result = generate_response_from_rag(query.strip())
        return jsonify(result)
    except Exception as e:
        # Avoid leaking secrets; return safe error.
        return jsonify({"error": "Server error while generating response.", "details": str(e)}), 500


if __name__ == "__main__":
    # Hugging Face Spaces (Docker) expects port 7860
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)
