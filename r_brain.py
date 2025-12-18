
"""Knowledge enhancer utilities: keyword extraction, web search, wikipedia.

Optional heavy dependencies (nltk, torch, wikipedia, googlesearch, textblob)
are imported with guards so the module can be linted and partially used
without every optional package installed.
"""

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None
import datetime

try:
    # googlesearch may be unavailable or installed under different package name
    from googlesearch import search
except Exception:
    search = None

try:
    import wikipedia
except Exception:
    wikipedia = None

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
except Exception:
    nltk = None
    word_tokenize = None
    sent_tokenize = None
    stopwords = None

# ========================
# 5. KNOWLEDGE ENHANCER
# ========================


class KnowledgeEnhancer:
    def __init__(self):
        self.search_cache = {}
        # If NLTK is unavailable, fall back to an empty stopword set
        try:
            self.stop_words = set(stopwords.words('english')) if stopwords is not None else set()
        except Exception:
            self.stop_words = set()

    def extract_keywords(self, text):
        """Use NLTK to extract important keywords from text"""
        tokens = word_tokenize(text.lower())

        # Remove stopwords and punctuation
        keywords = [token for token in tokens if token not in self.stop_words and token.isalnum()]

        # Keep only meaningful keywords (more than 2 characters)
        meaningful_keywords = [kw for kw in keywords if len(kw) > 2]

        return meaningful_keywords

    def web_search(self, query):
        """Enhanced search with keyword extraction"""
        try:
            keywords = self.extract_keywords(query)
            print(f"🔍 Keywords extracted: {keywords}")

            if not keywords:
                return "I couldn't identify clear keywords to search for."

            # Use keywords for more focused search
            search_query = " ".join(keywords[:3])  # Use top 3 keywords
            print(f"🔍 Searching for: {search_query}")

            if search is None:
                return "Search functionality not available (missing package)."

            try:
                urls = list(search(search_query, num_results=2, lang='en'))
            except Exception as e:
                return f"Search failed: {e}"

            if urls:
                return f"I found information about '{search_query}'. Check: {urls[0]}"
            return f"I couldn't find recent information about '{search_query}'."

        except Exception as e:
            return f"Search unavailable right now. Error: {e}"

    def wikipedia_lookup(self, topic):
        if wikipedia is None:
            return "Wikipedia lookup not available (missing package)."

        try:
            summary = wikipedia.summary(topic, sentences=2)
            return f"📖 {summary}"
        except Exception as e:
            return f"Couldn't find Wikipedia page for '{topic}': {e}"

    def get_current_time(self):
        now = datetime.datetime.now()
        return f"⏰ Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    def analyze_text_complexity(self, text):
        """Use NLTK to analyze text complexity"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0

        complexity = "simple" if avg_sentence_length < 10 else "complex"

        print(f"📊 Text Analysis: {len(sentences)} sentences, {len(words)} words")
        print(f"📊 Complexity: {complexity} (avg {avg_sentence_length:.1f} words/sentence)")

        return complexity
