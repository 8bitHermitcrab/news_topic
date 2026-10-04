from langdetect import detect


def detect_language(text):
    lang = detect(text)

    if lang == "ko":
        return "Korean"

    elif lang == "en":
        return "English"

    else:
        return "Unsupported"