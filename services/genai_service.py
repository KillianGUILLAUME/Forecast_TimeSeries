
from __future__ import annotations

import os
from typing import Final


_SYSTEM_PROMPT: Final[str] = (
    "You are an assistant specialized in economics and finance. "
    "Provide clear, concise answers that reference established economic concepts "
    "when relevant."
)


def fetch_economic_answer(question: str) -> str:
    """Call the OpenAI API to obtain an answer to *question*.

    Parameters
    ----------
    question:
        The economic question supplied by the user.

    Returns
    -------
    str
        Either the model response or an explanatory error message.
    """

    question = (question or "").strip()
    if not question:
        return "Veuillez saisir une question économique avant d'envoyer la requête."

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not api_key:
        return (
            "Aucune clé OpenAI détectée. Configurez la variable d'environnement "
            "OPENAI_API_KEY avant de poser une question."
        )

    try:
        import openai
    except Exception:  # pragma: no cover - dépend des installations utilisateur
        return (
            "Le module openai est introuvable. Installez le package 'openai' pour "
            "utiliser cette fonctionnalité."
        )

    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=600,
        )
    except Exception:  # pragma: no cover - dépend des réponses réseau/API
        return (
            "Impossible d'obtenir une réponse de l'API OpenAI pour le moment. "
            "Veuillez réessayer plus tard."
        )

    try:
        answer = response["choices"][0]["message"]["content"].strip()
    except Exception:  # pragma: no cover - format inattendu
        return "Réponse inattendue reçue depuis l'API OpenAI."

    return answer or "La réponse générée est vide."