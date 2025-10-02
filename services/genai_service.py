
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Final

from mistralai import Mistral

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())




_SYSTEM_PROMPT: Final[str] = (
    "You are an assistant specialized in economics and finance. "
    "Provide clear, concise answers that reference established economic concepts "
    "when relevant."
)
MODEL_PRIMARY = "magistral-small-2509"
MODEL_FALLBACK = "magistral-small-latest"


def to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    
    fragments: list[str] = []
    try:
        for part in content or []:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str):
                fragments.append(text)
    except Exception:
        pass

    if fragments:
        return "".join(fragments).strip()
    return str(content).strip()
    


def fetch_economic_answer(question: str) -> str:
    """Call the MistralAI API to obtain an answer to *question*.

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
        return "Veuillez saisir une question avant d'envoyer la requête."


    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return (
            "Aucune clé OpenAI détectée. Configurez la variable d'environnement "
            "OPENAI_API_KEY avant de poser une question."
            )
    


    client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    def _call(model_name: str) -> str:
    # Appel Mistral (synchrone)
        resp = client.chat.complete(
            model=model_name,
            messages=messages,
            temperature=0.7,   # optionnel
            max_tokens=600,     # optionnel

        )
        content = resp.choices[0].message.content
        text = to_text(content)
        return text or 'la réponse générée est vide.'
    
    try:
        return _call(MODEL_PRIMARY)
    except mistral_models.MistralError as e2:
        return(
            "Impossible d'obtenir une réponse pour le moment "
                f"(HTTP {getattr(e2, 'status_code', '?')}). "
                "Merci de réessayer plus tard."
        )
    except Exception:
        return 'Une erreur inattendue est survenue. Merci de réessayer plus tard.'
 