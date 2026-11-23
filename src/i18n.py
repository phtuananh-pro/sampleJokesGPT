"""Language-specific prompts and default file locations.

Topic IDs and humor style IDs are deliberately language-neutral, while the
visible prompt text is localized. Train and chat MUST use identical prompts:
the tokenizer learns their words as conditional context (FMg Part IV).
"""
from __future__ import annotations

from pathlib import Path

LANGUAGES = ("en", "fr", "vi")

GENERAL_PROMPTS = {
    "en": "Tell me a joke.",
    "fr": "Raconte-moi une blague.",
    "vi": "Kể cho tôi một chuyện cười.",
}

STYLE_PROMPTS = {
    "en": {
        "dad_joke": "Tell me a dad joke.",
        "pun": "Tell me a pun.",
        "one_liner": "Tell me a one-liner.",
        "wordplay": "Tell me a wordplay joke.",
    },
    "fr": {
        "dad_joke": "Raconte-moi une blague de papa.",
        "pun": "Raconte-moi un calembour.",
        "one_liner": "Raconte-moi une blague courte.",
        "wordplay": "Raconte-moi un jeu de mots.",
    },
    "vi": {
        "dad_joke": "Kể một câu chuyện cười kiểu bố.",
        "pun": "Kể một câu chơi chữ.",
        "one_liner": "Kể một câu chuyện cười ngắn.",
        "wordplay": "Kể một chuyện cười dựa trên chơi chữ.",
    },
}

TOPIC_PROMPT_TEMPLATES = {
    "en": "Tell me a joke about {topic}.",
    "fr": "Raconte-moi une blague sur {topic}.",
    "vi": "Kể cho tôi một chuyện cười về {topic}.",
}

FALLBACK_PREFIXES = {
    "en": "I couldn't find a good match, but here's another joke: ",
    "fr": "Je n'ai pas trouvé de blague qui corresponde, mais en voici une autre : ",
    "vi": "Tôi chưa tìm được chuyện cười đúng chủ đề, nhưng đây là một chuyện khác: ",
}

INCOMPLETE_MESSAGES = {
    "en": "I couldn't generate a complete joke. Try again.",
    "fr": "Je n'ai pas réussi à générer une blague complète. Réessaie.",
    "vi": "Tôi chưa tạo được một chuyện cười trọn vẹn. Hãy thử lại.",
}

SPEAKER_LABELS = {
    "en": ("You:", "Joke bot:"),
    "fr": ("Vous :", "Bot blagues :"),
    "vi": ("Bạn:", "Bot kể chuyện cười:"),
}


def data_path(language: str) -> Path:
    """English keeps the original filename to avoid breaking prior usage."""
    return Path("data/jokes_clean.jsonl" if language == "en"
                else f"data/jokes_clean_{language}.jsonl")


def topics_path(language: str) -> Path:
    """Return the taxonomy matching the selected corpus language."""
    return Path("data/topics.json" if language == "en"
                else f"data/topics_{language}.json")


def training_model_dir(language: str) -> Path:
    """Keep independent learned weights/tokenizers for every language."""
    return Path(f"models/jokes_{language}")


def chat_model_dir(language: str) -> Path:
    """Prefer a locally trained checkpoint, then a shipped runnable one.

    A model directory without model.pt is NOT runnable. English also supports
    an older default 'models/jokes' for existing users.
    """
    candidates = [training_model_dir(language),
                  Path(f"runnable_models/jokes_{language}")]
    if language == "en":
        candidates.append(Path("models/jokes"))
    for directory in candidates:
        if all((directory / file).is_file()
               for file in ("config.json", "tokenizer.json", "model.pt")):
            return directory
    # Error from load_model will make the expected path explicit.
    return candidates[0]
