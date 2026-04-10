from ollama import chat, ChatResponse
import json
import logging


logging.basicConfig(
    filename="OllamaUtils.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class OllamaUtils:
    def __init__(self, model):
        self.model = model

    def ask_ollama(self, prompt):
        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.message.content

    def ollama_true_false(self, prompt):
        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Only answer with 'yes' or 'no' in lowercase.\n" + prompt
                    ),
                }
            ],
        )

        answer = response.message.content.strip().lower()

        if answer == "yes":
            return True
        if answer == "no":
            return False

        return None

    def ollama_extract_word(self, text, dictionary):
        try:
            with open(dictionary, "r", encoding="utf-8") as f:
                data = json.load(f)

            keywords = []

            for category in data.values():
                if isinstance(category, list):
                    keywords.extend(category)

        except Exception as e:
            logging.error(f"[ERROR] Keyword loading failed: {e}")
            return None

        prompt = (
            "Extract exactly ONE keyword from the text.\n"
            "Return ONLY the keyword, no explanation.\n\n"
            f"Text: {text}\n"
            f"Allowed keywords: {keywords}\n"
        )

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        result = response.message.content.strip().lower()

        # Cleanup
        result = result.replace(".", "").replace(",", "")
        result = result.split()[0]

        return result
