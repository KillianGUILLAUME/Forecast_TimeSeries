
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Final



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
        from openai import OpenAI
    except Exception:
        OpenAI = None  # type: ignore[assignment]

    try:
        import openai
    except Exception:  # pragma: no cover - dépend des installations utilisateur
        return (
            "Le module openai est introuvable. Installez le package 'openai' pour "
            "utiliser cette fonctionnalité."
        )
        return (
            "Le module openai est introuvable. Installez le package 'openai' pour "
            "utiliser cette fonctionnalité."
        )

    client: Any | None = None

    if OpenAI is not None:  # SDK >= 1.0
        try:
            client = OpenAI(api_key=api_key)
        except Exception:  # pragma: no cover - dépend de l'installation utilisateur
            client = None

    if client is None:
        openai.api_key = api_key
        client = openai
        if not hasattr(client, "chat") and hasattr(client, "ChatCompletion"):
            client.chat = SimpleNamespace(  # type: ignore[attr-defined]
                completions=SimpleNamespace(
                    create=lambda **kwargs: client.ChatCompletion.create(**kwargs)  # type: ignore[attr-defined]
                )
            )

    response = None

    if hasattr(client, "responses"):
        try:
            response = client.responses.create(  # type: ignore[call-arg]
                model="gpt-3.5-turbo",
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_output_tokens=600,
            )
        except Exception:  # pragma: no cover - dépend des réponses réseau/API
            response = None

    if response is None and hasattr(client, "chat"):
        chat = getattr(client, "chat")
        completions = getattr(chat, "completions", None)
        create = getattr(completions, "create", None)
        if callable(create):
            try:
                response = create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.3,
                    max_tokens=600,
                )
            except Exception:  # pragma: no cover - dépend des réponses réseau/API
                response = None

    if response is None:
        return (
            "Impossible d'accéder aux API de génération OpenAI avec le SDK installé. "
            "Vérifiez que votre version du package 'openai' supporte l'endpoint "
            "responses ou chat.completions."
        )

    answer = _extract_text_from_response(response)
    if answer is None:
        return answer or "La réponse générée est vide."



def _extract_text_from_response(response: Any) -> str | None:
    """Normalise les différents formats de réponse OpenAI en chaîne de caractères."""

    if response is None:
        return None

    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    fragments: list[str] = []
    if output:
        for item in output:
            content_list = getattr(item, "content", None) or []
            for content in content_list:
                segment = getattr(content, "text", None)
                if hasattr(segment, "value"):
                    segment = getattr(segment, "value")
                if isinstance(segment, str):
                    fragments.append(segment)
    if fragments:
        return "".join(fragments).strip()

    data = response
    if hasattr(response, "model_dump") and callable(getattr(response, "model_dump")):
        try:
            data = response.model_dump()
        except Exception:  # pragma: no cover - dépend du SDK
            data = response

    if isinstance(data, dict):
        try:
            return str(
                data["choices"][0]["message"]["content"]  # type: ignore[index]
            ).strip()
        except Exception:  # pragma: no cover - format inattendu
            return None

    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
        if isinstance(choices, list) and choices:
            message = choices[0]
            if hasattr(message, "message"):
                message = getattr(message, "message")
            content = None
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()

    return None