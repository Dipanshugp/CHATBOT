# ------------------ IMPORTS ------------------
import json, random, pickle, wikipedia, re, time, sys
import numpy as np
from colorama import Fore, init
init()

grn = Fore.GREEN
blu = Fore.BLUE

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download nltk data (safe)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# ------------------ LOAD FILES ------------------
try:
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
    model = load_model('IntellichatModel.h5')
except Exception as e:
    print("Error loading files:", e)
    sys.exit()

# ------------------ NLP ------------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    bag = [0] * len(words)
    for s in clean_up_sentence(sentence):
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints):
    if not ints:
        return "Sorry, I didn't understand."

    tag = ints[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# ------------------ WIKIPEDIA ------------------
def extract_subject(question):
    question = re.sub(r'[^\w\s]', '', question)
    words = question.split()
    return ' '.join(words[2:])

def search_wikipedia(question):
    try:
        subject = extract_subject(question)
        return wikipedia.summary(subject, sentences=2)
    except:
        return "Sorry, I couldn't find information."

# ------------------ LOGIC ------------------
def process_message(message):
    # Arithmetic
    if any(op in message for op in ['+', '-', '*', '/', '**']):
        try:
            return f"Answer is {eval(message)}"
        except:
            return "Error in calculation!"

    # Wikipedia
    elif message.lower().startswith(('who','what','where','which','when','how')):
        return search_wikipedia(message)

    # Chatbot
    else:
        ints = predict_class(message)
        return get_response(ints)

# ------------------ CHAT LOOP ------------------
def chat():
    print("Chatbot is running (type 'quit')")
    while True:
        try:
            print(grn, end='')
            message = input("User: ")

            if message.lower() in ['quit','exit','close']:
                break

            response = process_message(message)
            print(blu + "IntelliChat:", response)

        except EOFError:
            print("\nCI mode detected. Exiting...")
            break

# ------------------ MAIN ------------------
if __name__ == "__main__":
    if sys.stdin.isatty():
        chat()
    else:
        print("Chatbot loaded successfully (CI mode)")
