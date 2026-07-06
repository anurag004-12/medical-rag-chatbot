import re

# Greeting words
GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you"
}

# Thank you words
THANKS = {
    "thanks",
    "thank you",
    "thankyou"
}

# Bye words
FAREWELLS = {
    "bye",
    "goodbye",
    "see you",
    "take care"
}


def detect_intent(query: str):

    # Convert to lowercase
    query = query.lower().strip()

    # Remove punctuation
    query = re.sub(r"[^\w\s]", "", query)

    if query in GREETINGS:
        return "greeting"

    elif query in THANKS:
        return "thanks"

    elif query in FAREWELLS:
        return "farewell"

    else:
        return "medical"