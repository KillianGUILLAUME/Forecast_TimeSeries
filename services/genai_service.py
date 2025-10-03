
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Final

from mistralai import Mistral, MistralError
# from mistralai.exceptions import MistralException

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())




_SYSTEM_PROMPT: Final[str] = (
    "You are an assistant specialized in economics and finance.\n\n"
    "Structure every response as follows:\n"
    "- Executive numeric summary that highlights the main figures and deltas.\n"
    "- Key indicators with concise commentary (inflation, GDP, rates, markets, etc.).\n"
    "- Analytical insights linking data to economic or financial mechanisms.\n"
    "- Forward-looking scenarios or outlook, explicitly stating assumptions.\n"
    "- Sources or data references enabling verification.\n\n"
    "Only produce economic or financial analyses; politely decline any other request."
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
            "Aucune clé Mistral AI détectée. Configurez la variable d'environnement "
            "MISTRAL_API_KEY avant de poser une question."
            )
    


    client = Mistral(api_key=api_key)

    metadata_message = {
        "role": "system",
        "content": (
            "Metadata | date: "
            f"{datetime.now(timezone.utc).isoformat()}Z ; source: Forecast_TimeSeries service"
        ),
    }


    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        metadata_message,
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
    
    last_mistral_error: MistralError | None = None
    last_unexpected_error: Exception | None = None

    for model_name in (MODEL_PRIMARY, MODEL_FALLBACK):
        try:
            return _call(model_name)
        except MistralError as error:
            last_mistral_error = error
        except Exception as error:  # pragma: no cover - unexpected, allows retry
            last_unexpected_error = error

    if last_mistral_error is not None:
        status_code = getattr(last_mistral_error, "status_code", "?")
        return (
            "Impossible d'obtenir une réponse du service Mistral pour le moment "
            f"(HTTP {status_code}). Merci de réessayer plus tard."
        )
    
    if last_unexpected_error is not None:
        return (
            "Une erreur inattendue est survenue lors de l'appel au service Mistral. "
            "Merci de réessayer plus tard."
        )
    return "Une erreur inattendue est survenue. Merci de réessayer plus tard."
