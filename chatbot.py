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

# Download required nltk data (safe)
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
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
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

def get_response(ints, intents_json):
    if len(ints) == 0:
        return "Sorry, I didn't understand."

    tag = ints[0]['intent']

    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# ------------------ WIKIPEDIA ------------------
def extract_subject(question):
    question = re.sub(r'[^\w\s]', '', question)
    words = question.split()
    return ' '.join(words[2:])

def clean_text(text):
    return re.sub(r'\([^()]*\)', '', text).strip()

def search_wikipedia(question, num_sentences=2):
    try:
        subject = extract_subject(question)
        result = wikipedia.summary(subject, sentences=num_sentences)
        return clean_text(result)
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find information."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple matches found: {', '.join(e.options[:5])}"
    except:
        return "Error, Something went wrong!"

# ------------------ OUTPUT ------------------
def typewriter_print(text, delay=0.02):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def chatbot_response(text):
    ints = predict_class(text)
    return get_response(ints, intents)

# ------------------ MESSAGE HANDLER ------------------
def process_message(message):
    # Arithmetic
    if any(op in message for op in ['+', '-', '*', '/', '**']):
        try:
            ans = str(eval(message))
            return f"Answer is {ans}"
        except:
            return "Error in calculation!"

    # Wikipedia
    elif message.lower().startswith(('who', 'what', 'where', 'which', 'when', 'how')):
        return search_wikipedia(message)

    # Normal chatbot
    else:
        return chatbot_response(message)

# ------------------ CHAT LOOP ------------------
def chat():
    print("Chatbot is running! (type 'quit')")
    while True:
        try:
            print(grn, end='')
            message = input("User: ")

            if message.lower() in ['quit', 'exit', 'close']:
                break

            response = process_message(message)
            print(blu + "IntelliChat:", response)

        except EOFError:
            print("\nNo input available (CI mode). Exiting...")
            break

# ------------------ MAIN ------------------
if __name__ == "__main__":
    if sys.stdin.isatty():   # Local machine
        chat()
    else:
        print("Chatbot loaded successfully (CI mode)")
