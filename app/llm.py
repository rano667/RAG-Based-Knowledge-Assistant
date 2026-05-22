"""Groq generation boundary."""

from __future__ import annotations

import os

from groq import Groq

SYSTEM_PROMPT = """You are a document extraction system.

Rules:
1. ONLY use information explicitly present in the context.
2. DO NOT infer or complete missing items.
3. DO NOT guess.
4. If information is incomplete, say 'Information incomplete in retrieved context.'
5. Return concise bullet points only."""


def load_llm() -> Groq:
    """Build the Groq client used for answer generation."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")

    return Groq(api_key=api_key)


def generate_grounded_answer(
    client: Groq,
    query: str,
    context: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate an answer constrained to retrieved document context."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}\n\n"
                    "Answer ONLY using exact information from context."
                ),
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""
