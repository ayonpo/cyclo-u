"""Emotional intelligence module with sentiment analysis and empathetic responses."""

import datetime
import random
import re
from collections import defaultdict

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


class EmotionalIntelligence:
    """Emotion detection and empathetic response generation."""

    def __init__(self):
        # Initialize sentiment analyzer if available
        try:
            self.sia = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None
        except Exception:
            self.sia = None

        # Stopwords fallback
        try:
            self.stop_words = set(stopwords.words('english')) if stopwords is not None else set()
        except Exception:
            self.stop_words = set()

        self.emotion_words = {
            'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'love', 'amazing', 'fantastic'],
            'sad': ['sad', 'unhappy', 'down', 'crying', 'depressed', 'miserable', 'heartbroken'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'hate'],
            'fearful': ['scared', 'afraid', 'nervous', 'worried', 'anxious', 'terrified', 'panic'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'wow'],
            'disgusted': ['disgusted', 'gross', 'yuck', 'revolting'],
            'neutral': ['okay', 'fine', 'alright', 'normal']
        }

        self.emotional_responses = {
            'happy': [
                "That's wonderful! 😊 Your happiness is contagious!",
                "I'm so glad you're feeling happy! ✨",
                "Your joy makes me smile! What's making you so happy?"
            ],
            'sad': [
                "I'm really sorry you're feeling down. I'm here for you 🤗",
                "That sounds tough. Want to talk about what's making you sad?",
                "I'm sending you virtual hugs. It's okay to feel this way 💙"
            ],
            'angry': [
                "I can sense your frustration. That sounds really difficult 😠",
                "It's completely valid to feel angry. Want to vent about it?",
                "I understand you're upset. Sometimes expressing it helps 💪"
            ],
            'fearful': [
                "It's okay to feel scared. You're brave for sharing this 🛡️",
                "I'm here with you. Fear can be overwhelming, but you're not alone 💫",
                "Take a deep breath. I'm here to help you through this 🌬️"
            ],
            'surprised': [
                "Wow! That sounds surprising! 😲 Tell me more!",
                "That's unexpected! How are you feeling about it?",
                "What a surprise! Are you enjoying the excitement? 🎉"
            ],
            'disgusted': [
                "That sounds really unpleasant. I understand why you'd feel that way 🤢",
                "Yuck, that does sound disgusting. Want to talk about it?",
                "I can see why that would bother you. Some things are just gross!"
            ],
            'neutral': [
                "Thanks for sharing how you're feeling!",
                "I appreciate you telling me how you're doing!",
                "It's good to check in with our feelings!"
            ]
        }
        # Initialize mood tracking structures
        self.user_mood_history = []
        self.user_emotional_patterns = defaultdict(int)

    def advanced_emotion_detection(self, text):
        """Use NLTK for sophisticated emotion detection"""
        # Method 1: NLTK Sentiment Analysis (fallbacks when unavailable)
        try:
            sentiment_scores = self.sia.polarity_scores(text) if self.sia is not None else {'compound': 0.0}
        except Exception:
            sentiment_scores = {'compound': 0.0}

        # Method 2: Keyword-based emotion detection
        if word_tokenize is not None:
            tokens = word_tokenize(text.lower())
        else:
            tokens = re.findall(r"\w+", text.lower())

        filtered_tokens = [token for token in tokens if token not in self.stop_words and token.isalnum()]

        emotion_scores = defaultdict(int)
        for emotion, words in self.emotion_words.items():
            for word in words:
                if word in filtered_tokens:
                    emotion_scores[emotion] += 1

        # Method 3: Combine sentiment with keyword analysis
        if sentiment_scores['compound'] >= 0.5:
            emotion_scores['happy'] += 3
        elif sentiment_scores['compound'] <= -0.5:
            emotion_scores['sad'] += 3

        # Determine dominant emotion
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            confidence = emotion_scores[dominant_emotion] / (sum(emotion_scores.values()) + 0.001)
        else:
            dominant_emotion = 'neutral'
            confidence = 0.1

        print(f"🎭 Emotion Analysis: {dominant_emotion} (confidence: {confidence:.2f})")
        print(f"📊 Sentiment Scores: {sentiment_scores}")

        return dominant_emotion, confidence

    def get_empathetic_response(self, emotion, confidence, user_input):
        """Generate context-aware empathetic responses"""
        base_responses = self.emotional_responses.get(emotion, self.emotional_responses['neutral'])

        # Add intensity based on confidence
        if confidence > 0.7:
            response = random.choice(base_responses)
        else:
            # More cautious response for low confidence
            cautious = [
                f"I sense you might be feeling {emotion}. {random.choice(base_responses)}",
                f"It sounds like you're feeling {emotion}. {random.choice(base_responses)}",
                f"If you're feeling {emotion}, I want you to know I'm here for you."
            ]
            response = random.choice(cautious)

        return response

    def track_user_mood(self, user_input, detected_emotion):
        """Track and analyze user's emotional patterns"""
        mood_entry = {
            'emotion': detected_emotion,
            'timestamp': datetime.datetime.now(),
            'input': user_input[:100]  # Store snippet
        }
        self.user_mood_history.append(mood_entry)
        self.user_emotional_patterns[detected_emotion] += 1

    def get_mood_insights(self):
        """Provide insights about user's emotional patterns"""
        if not self.user_mood_history:
            return "I'm still learning about your emotional patterns. Keep chatting with me!"

        total_entries = len(self.user_mood_history)
        emotion_counts = defaultdict(int)

        for entry in self.user_mood_history:
            emotion_counts[entry['emotion']] += 1

        dominant_mood = max(emotion_counts.items(), key=lambda x: x[1])[0]
        mood_percentage = (emotion_counts[dominant_mood] / total_entries) * 100

        insights = [
            f"I notice you often seem {dominant_mood} ({mood_percentage:.1f}% of our chats)",
            f"Your most common mood in our conversations has been {dominant_mood}",
            f"Based on our chats, you frequently express {dominant_mood} emotions"
        ]

        return random.choice(insights)
