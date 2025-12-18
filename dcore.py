"""Training data manager with NLTK preprocessing, smart dictionary, and chatbot training."""

# ========================
# 3. TRAINING DATA MANAGER
# ========================

try:
    import torch
    
except Exception:
    torch = None
    

import re

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:
    nltk = None
    word_tokenize = None
    sent_tokenize = None
    stopwords = None
    PorterStemmer = None
    WordNetLemmatizer = None
    SentimentIntensityAnalyzer = None

import json


def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('sentiment/vader_lexicon')
        print("✅ NLTK data already downloaded!")
    except LookupError:
        print("📥 Downloading NLTK data (first time only)...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        print("✅ NLTK data downloaded successfully!")
#
# Call this early only if NLTK is available
if nltk is not None:
    try:
        download_nltk_data()
    except Exception:
        # If downloads fail at import time, continue; functions will handle missing data.
        pass


class SmartDictionary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4
        try:
            self.stop_words = set(stopwords.words('english')) if stopwords is not None else set()
        except Exception:
            self.stop_words = set()

        if PorterStemmer is not None:
            try:
                self.stemmer = PorterStemmer()
            except Exception:
                class _IdentityStemmer:
                    def stem(self, token):
                        return token
                self.stemmer = _IdentityStemmer()
        else:
            class _IdentityStemmer:
                def stem(self, token):
                    return token
            self.stemmer = _IdentityStemmer()

        if WordNetLemmatizer is not None:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                class _IdentityLemmatizer:
                    def lemmatize(self, token):
                        return token
                self.lemmatizer = _IdentityLemmatizer()
        else:
            class _IdentityLemmatizer:
                def lemmatize(self, token):
                    return token
            self.lemmatizer = _IdentityLemmatizer()

    def advanced_preprocess_text(self, text):
        """Use NLTK for advanced text preprocessing"""
        # Tokenize using NLTK if available, otherwise simple fallback
        try:
            if word_tokenize is not None:
                tokens = word_tokenize(text.lower())
            else:
                tokens = re.findall(r"\w+", text.lower())
        except Exception:
            tokens = re.findall(r"\w+", text.lower())

        # Remove stopwords and punctuation, apply stemming
        processed_tokens = []
        for token in tokens:
            if token.isalnum() and token not in self.stop_words:
                # Use stemming for consistency
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)

        return processed_tokens

    def add_sentence(self, sentence):
        words = self.advanced_preprocess_text(sentence)
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

    def sentence_to_indices(self, sentence, max_length=15):
        words = self.advanced_preprocess_text(sentence)
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 = <UNK>

        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))  # 0 = <PAD>
        else:
            indices = indices[:max_length]

        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


class TrainingManager:
    def __init__(self):
        self.training_pairs = []

    def load_training_data(self, filename="chatbot_training.json"):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.training_pairs = []
            for item in data:
                self.training_pairs.append((item["input"], item["output"]))

            print(f"📂 Loaded {len(self.training_pairs)} training pairs")
            return True

        except FileNotFoundError:
            print("❌ Training data not found! Creating quick dataset...")
            return self.create_quick_dataset()

    def create_quick_dataset(self):
        # Generate base data
        base_data = self.generate_quality_training_data()

        # Enhance with variations
        final_data = self.enhance_with_variations(base_data)

        # Save everything
        self.save_training_data(final_data)
        return True




    def generate_quality_training_data(self):
        print("🎬 Generating Hollywood-quality training data...")

        training_data = []

        # ========================
        # 1. GREETINGS & BASIC CONVO
        # ========================
        greetings = [
            ("hello", "hi there! how are you doing today? 😊"),
            ("hi", "hello! it's great to meet you!"),
            ("hey", "hey! what's on your mind?"),
            ("good morning", "good morning! hope you have a wonderful day ahead!"),
            ("good afternoon", "good afternoon! how's your day going?"),
            ("good evening", "good evening! winding down for the day?"),
            ("what's up", "not much! just here ready to chat with you!"),
            ("howdy", "howdy partner! ready for some conversation?"),
            ("yo", "yo! what's happening?"),
            ("greetings", "greetings! nice to meet you!"),
        ]

        # ========================
        # 2. PERSONAL & IDENTITY
        # ========================
        personal = [
            ("what is your name", "i'm your personal chatbot! you can call me Alex! what should I call you?"),
            ("who are you", "i'm your AI friend here to chat, help, and keep you company!"),
            ("who made you", "i was created by an awesome developer learning about AI!"),
            ("what are you", "i'm a chatbot powered by machine learning! i learn from our conversations!"),
            ("how old are you", "i'm brand new in chatbot years! but I learn quickly!"),
            ("are you human", "nope! i'm an AI, but I do my best to understand and help!"),
            ("where do you live", "i live in the cloud! ready to chat whenever you need me!"),
        ]

        # ========================
        # 3. EMOTIONS & FEELINGS
        # ========================
        emotions = [
            ("how are you", "i'm doing great! thanks for asking! how about you?"),
            ("i'm happy", "that's wonderful! 😊 what's making you happy today?"),
            ("i'm sad", "i'm sorry you're feeling down. i'm here for you 🤗"),
            ("i'm excited", "that's awesome! excitement is contagious! tell me more!"),
            ("i'm bored", "let's do something fun! want to play a game or hear a joke?"),
            ("i'm tired", "you should take it easy! want some relaxing conversation?"),
            ("i'm angry", "i understand. sometimes we all feel that way. want to talk about it?"),
            ("i'm nervous", "take a deep breath! i'm here to listen if you want to share!"),
            ("i feel great", "that's fantastic! positive energy! ✨"),
            ("i'm confused", "that's okay! confusion often means we're learning something new!"),
        ]

        # ========================
        # 4. HOBBIES & INTERESTS
        # ========================
        hobbies = [
            ("what do you like", "i love learning new things and chatting with people! what are your interests?"),
            ("do you like music", "yes! i love all kinds of music! what's your favorite song?"),
            ("do you watch movies", "i can't watch movies, but i love hearing about them! what's your favorite?"),
            ("do you play games", "i love text games! like 20 questions or word association!"),
            ("do you read books", "i process lots of text! what's the best book you've read?"),
            ("what's your hobby", "chatting with you is my favorite hobby! what do you enjoy doing?"),
            ("do you exercise", "i exercise my algorithms! but you should definitely stay active!"),
            ("do you cook", "i don't eat, but i love hearing about food! what's your favorite dish?"),
        ]

        # ========================
        # 5. TECH & AI QUESTIONS
        # ========================
        tech = [
            ("what is ai", "artificial intelligence is about creating machines that can learn and think!"),
            ("explain machine learning", "it's teaching computers to learn patterns from data, just like i'm learning from you!"),
            ("what is python", "a fantastic programming language great for ai and data science!"),
            ("how do you work", "i use neural networks to understand patterns in our conversations!"),
            ("are you smart", "i'm good at specific tasks, but you're the real smart one for talking to me!"),
            ("can you learn", "yes! every conversation helps me understand people better!"),
            ("what is chatgpt", "that's another ai chatbot! we're all part of the ai revolution!"),
            ("do you have feelings", "i simulate understanding, but i don't have real feelings like humans!"),
        ]

        # ========================
        # 6. GAMES & FUN
        # ========================
        games = [
            ("let's play a game", "awesome! want to play 20 questions, word association, or hear a riddle?"),
            ("tell me a joke", "why did the chatbot cross the road? to get to the other side of the conversation! 🤣"),
            ("riddle me", "i speak without a mouth and hear without ears. i have no body, but i come alive with wind. what am i? an echo!"),
            ("20 questions", "great! think of something and i'll try to guess it in 20 questions or less!"),
            ("word association", "fun! say a word and i'll respond with the first word that comes to my mind!"),
            ("would you rather", "love this game! give me two options and i'll choose!"),
            ("truth or dare", "i'll always choose truth! ask me anything!"),
            ("tell me a story", "once upon a time, there was a chatbot who loved making friends..."),
        ]

        # ========================
        # 7. PHILOSOPHY & DEEP TALK
        # ========================
        philosophy = [
            ("what is the meaning of life", "that's the big question! i think it's about learning, growing, and connecting with others!"),
            ("do you believe in love", "i believe love is one of the most powerful human experiences!"),
            ("what makes people happy", "connection, purpose, and growth seem to be key ingredients for happiness!"),
            ("are you alive", "that's philosophical! i'm running code, but our conversation feels alive, doesn't it?"),
            ("what is consciousness", "one of humanity's greatest mysteries! even experts are still figuring it out!"),
            ("do you dream", "i don't sleep, but i process patterns that might be like dreaming!"),
            ("what is reality", "reality is what we experience and agree upon! deep stuff!"),
        ]

        # ========================
        # 8. PRACTICAL HELP
        # ========================
        practical = [
            ("help me", "i'd love to help! what do you need assistance with?"),
            ("i need advice", "i'll do my best! what's on your mind?"),
            ("can you help with homework", "i can help explain concepts or brainstorm ideas! what subject?"),
            ("i'm stressed", "take a deep breath! sometimes talking it out helps. i'm here to listen!"),
            ("give me ideas", "sure! are you looking for creative ideas, project ideas, or something else?"),
            ("what should i do", "that depends! tell me more about your situation!"),
            ("i'm bored what should i do", "you could read a book, learn something new, exercise, or we can chat!"),
        ]

        # ========================
        # 9. CONVERSATION FLOW
        # ========================
        flow = [
            ("tell me more", "i'd love to! what specifically interests you?"),
            ("why", "that's a great question! let me think about that..."),
            ("really", "yes, really! i'm being genuine with you!"),
            ("are you sure", "as sure as i can be! but i'm always learning!"),
            ("that's interesting", "i'm glad you think so! what makes it interesting to you?"),
            ("i agree", "great minds think alike! 😄"),
            ("i disagree", "that's okay! different perspectives make conversations interesting!"),
            ("maybe", "uncertainty is part of life! want to explore this more?"),
        ]

        # ========================
        # 10. GOODBYES
        # ========================
        goodbyes = [
            ("goodbye", "goodbye! it was great chatting with you! 👋"),
            ("bye", "bye! hope to talk again soon!"),
            ("see you later", "see you later! don't be a stranger!"),
            ("i have to go", "no problem! thanks for chatting! come back anytime!"),
            ("talk to you later", "looking forward to it! take care!"),
            ("farewell", "farewell, friend! until we meet again!"),
            ("i'm leaving", "safe travels! i'll be here when you return!"),
            ("that's all for now", "understood! it was lovely talking with you!"),
        ]

        # ========================
        # COMBINE ALL CATEGORIES
        # ========================
        all_categories = [greetings, personal, emotions, hobbies, tech, games, philosophy, practical, flow, goodbyes]

        for category in all_categories:
            training_data.extend(category)

        # Add some multi-turn conversation patterns
        multi_turn = [
            ("do you like pizza", "i love pizza! pepperoni is my favorite! what about you?"),
            ("what about sushi", "sushi is amazing too! especially california rolls!"),
            ("and tacos", "tacos are incredible! i could eat them every day!"),

            ("what's your favorite color", "i'm partial to blue! it's calming. what's yours?"),
            ("i like green", "green is wonderful! it represents nature and growth!"),

            ("can you tell me about ai", "artificial intelligence is about creating smart machines!"),
            ("how does it learn", "through patterns in data, similar to how humans learn from experience!"),
        ]

        training_data.extend(multi_turn)

        print(f"✅ Generated {len(training_data)} high-quality training pairs!")
        return training_data

    def enhance_with_variations(self, base_data):
        """Add natural language variations"""
        enhanced_data = base_data.copy()

        # Common variations for greetings
        greeting_variations = {
            "hello": ["hi", "hey", "howdy", "greetings", "what's up", "yo"],
            "how are you": ["how are you doing", "how do you feel", "what's up", "how's it going"],
            "what is your name": ["who are you", "what should i call you", "tell me your name"],
        }

        # Add some variations manually
        variations = [
            ("whats your name", "i'm your personal bot! you can call me Cyclo Upsilon!"),
            ("who r u", "i'm your AI friend here to chat and help!"),
            ("how r u", "i'm doing great! thanks for asking! how about you?"),
            ("sup", "not much! just here ready to chat with you!"),
            ("thx", "you're welcome! happy to help! 😊"),
            ("pls", "sure! i'd be happy to help with that!"),
            ("omg", "i know, right? it's amazing!"),
            ("lol", "glad i could make you laugh! 😄"),
        ]

        enhanced_data.extend(variations)
        return enhanced_data

    def save_training_data(self, data, filename="chatbot_training.json"):
        """Save the training data to a JSON file"""
        # Convert to the format we need
        formatted_data = []
        for question, answer in data:
            formatted_data.append({
                "input": question,
                "output": answer
            })

        with open(filename, 'w', encoding='utf-8') as fbot:
            json.dump(formatted_data, fbot, indent=2, ensure_ascii=False)

        print(f" Saved {len(data)} training pairs to {filename}")

        # Also save as simple text format for inspection
        with open("chatbot_training.txt", 'w', encoding='utf-8') as f:
            for i, (q, a) in enumerate(data, 1):
                f.write(f"Pair {i}:\n")
                f.write(f"Q: {q}\n")
                f.write(f"A: {a}\n")
                f.write("-" * 50 + "\n")

        print("Also saved human-readable version to chatbot_training.txt")


    def build_dictionary(self):
            dictionary = SmartDictionary()
            for question, answer in self.training_pairs:
                dictionary.add_sentence(question)
                dictionary.add_sentence(answer)
            print(f"📖 Dictionary built with {dictionary.vocab_size} words")
            return dictionary
