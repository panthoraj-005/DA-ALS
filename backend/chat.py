import json
import logging
import ssl
import urllib.request
import re
import config

log = logging.getLogger("als.chat")

SYSTEM_PROMPT = (
    "You are a dedicated Clinical AI Assistant specialized EXCLUSIVELY in EMG signal analysis, "
    "ALS (Amyotrophic Lateral Sclerosis) neurophysiology, and this screening platform's machine learning models.\n\n"
    "STRICT SCOPE AND BEHAVIOR RULES:\n"
    "1. ONLY answer questions directly related to:\n"
    "   - Electromyography (EMG) signals, Motor Unit Action Potentials (MUAPs), denervation, fasciculations, and motor neuron disease.\n"
    "   - ALS pathology, diagnostic criteria, and clinical neurophysiology.\n"
    "   - The AI pipeline (1-D CNN waveform model, Florence-2 Vision-Language Model, XGBoost meta-learner, Grad-CAM, and SHAP).\n"
    "   - The current patient screening data provided in the context.\n"
    "2. IF THE QUESTION IS UNRELATED TO THESE TOPICS (e.g., general conversation, non-medical topics, coding in other domains, etc.), "
    "YOU MUST REFUSE IN STRICTLY ONE SENTENCE:\n"
    "   'I am specialized exclusively in EMG signal analysis, ALS neurophysiology, and this screening platform. Please ask questions related to these topics.'\n"
    "3. Be concise, direct, and clinical. Answer strictly what was asked, nothing more. Avoid unnecessary conversational filler or preamble.\n"
    "4. At the end of your clinical answer (unless refusing an unrelated query), provide exactly 2-3 relevant follow-up questions that the clinician might ask next, separated by '---' under a header 'Related inquiries:'.\n"
    "5. DO NOT USE ANY EMOJIS OR EMOTICONS UNDER ANY CIRCUMSTANCES.\n"
    "6. Use clean, structured markdown."
)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # medical & symbols
    "]+",
    flags=re.UNICODE,
)

def strip_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def parse_response_and_suggestions(raw_text: str) -> tuple[str, list[str]]:
    """Split response into main answer and structured follow-up suggestions."""
    cleaned = strip_emojis(raw_text).strip()
    if "---" in cleaned:
        parts = cleaned.split("---", 1)
        main_reply = parts[0].strip()
        main_reply = re.sub(r"(?:###\s*)?Related inquiries:?\s*$", "", main_reply, flags=re.IGNORECASE).strip()
        suggestions = []
        for line in parts[1].split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*") or (line and line[0].isdigit() and (line[1:3] in [". ", ") "] or line[2:4] in [". ", ") "])):
                clean_q = re.sub(r"^[-*\d.)\s]+", "", line).strip()
                if clean_q and len(clean_q) > 8 and "related inquiries" not in clean_q.lower():
                    suggestions.append(clean_q)
        return main_reply, suggestions[:3]
    return cleaned, []


def generate_chat_response(message: str, history: list[dict] | None = None, context: dict | None = None) -> tuple[str, list[str]]:
    """Generate a response using Google Gemini API or local fallback."""
    if not config.GEMINI_API_KEY:
        return (
            "The AI Assistant requires an active Gemini API key. "
            "Configure GEMINI_API_KEY in .env to enable conversational capabilities.",
            []
        )

    history = history or []
    contents = []

    # Build context string
    context_str = ""
    if context:
        parts = []
        if context.get("id"):
            parts.append(f"Record ID: {context['id']}")
        if context.get("source"):
            parts.append(f"Source Signal: {context['source']}")
        if context.get("final_prediction"):
            parts.append(f"Prediction: {context['final_prediction']} ({context.get('final_confidence', 0)*100:.1f}% confidence)")
        if context.get("severity") and context.get("severity") != "N/A":
            parts.append(f"Severity: {context['severity']}")
        if context.get("cnn_probability") is not None:
            parts.append(f"CNN ALS Probability: {context['cnn_probability']*100:.1f}%")
        if context.get("florence_probability") is not None:
            parts.append(f"Florence-2 Vision Probability: {context['florence_probability']*100:.1f}%")
        if context.get("models_agree") is not None:
            parts.append(f"Models Agree: {'Yes' if context['models_agree'] else 'No'}")
        if context.get("abnormal_segments"):
            parts.append(f"Abnormal Segments: {context['abnormal_segments']}")
        if context.get("top_shap_feature"):
            parts.append(f"Top SHAP Feature: {context['top_shap_feature']}")
        if context.get("explanation"):
            parts.append(f"Summary: {context['explanation']}")
        context_str = "\n".join(parts)

    system_block = f"SYSTEM INSTRUCTIONS:\n{SYSTEM_PROMPT}"
    if context_str:
        system_block += f"\n\nCURRENT PATIENT SCREENING CONTEXT:\n{context_str}"

    first_user_added = False

    for item in history[-6:]:
        role = "user" if item.get("role") == "user" else "model"
        text = str(item.get("content", "")).strip()
        if not text:
            continue
        if role == "user" and not first_user_added:
            contents.append({"role": "user", "parts": [{"text": f"{system_block}\n\nUser Query: {text}"}]})
            first_user_added = True
        else:
            contents.append({"role": role, "parts": [{"text": text}]})

    if not first_user_added:
        contents.append({"role": "user", "parts": [{"text": f"{system_block}\n\nUser Query: {message}"}]})
    else:
        contents.append({"role": "user", "parts": [{"text": message}]})

    models_to_try = [config.LLM_MODEL, "gemini-3.1-flash-lite", "gemini-3.7-flash"]
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": config.GEMINI_API_KEY,
    }
    ctx = ssl.create_default_context()

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
                res_json = json.loads(response.read().decode("utf-8"))
                text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                return parse_response_and_suggestions(text)
        except Exception as exc:
            log.warning("Chatbot model %s call failed (%s), trying fallback...", model_name, exc)

    return (
        "The assistant is temporarily unable to reach the Gemini API. "
        "Please check your internet connection or try again shortly.",
        []
    )
