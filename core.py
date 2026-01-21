"""Complete AI chatbot with NLTK integration, emotional intelligence, and knowledge enhancement."""

import datetime
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

try:
    from nltk.tokenize import word_tokenize
except Exception:
    word_tokenize = None

# First-party/local imports
from cunn import EnhancedChatBrain
from dcore import TrainingManager, SmartDictionary
from l_brain import EmotionalIntelligence
from r_brain import KnowledgeEnhancer
import comms as sl

# Note: `SmartDictionary` and `TrainingManager` implementations are provided
# by `dcore.py`. We import them from there to avoid duplication.


class CompleteAIChatbot:
    def __init__(self):
        self.hidden_size = 128
        self.embedding_dim = 64
        self.trainer = TrainingManager()
        self.dictionary = None
        self.model = None
        self.ei_system = EmotionalIntelligence()
        self.knowledge_enhancer = KnowledgeEnhancer()
        self.conversation_memory = deque(maxlen=50)

    def setup(self):
        print("🔄 Setting up your AI chatbot with NLTK...")

        if not self.trainer.load_training_data():
            return False

        # Use NLTK-enhanced dictionary
        self.dictionary = SmartDictionary()
        for question, answer in self.trainer.training_pairs:
            self.dictionary.add_sentence(question)
            self.dictionary.add_sentence(answer)

        print(f"📖 NLTK Dictionary built with {self.dictionary.vocab_size} words")

        self.model = EnhancedChatBrain(self.dictionary.vocab_size, self.hidden_size, self.embedding_dim)

        print("✅ AI with NLTK setup complete!")
        return True


    def train(self, epochs=100, learning_rate=0.01):
        if self.model is None:
            return

        print("🎯 Training AI...")

        # Prepare training data

        X_train = []
        y_train = []

        for question, answer in self.trainer.training_pairs:
            q_indices = self.dictionary.sentence_to_indices(question)
            a_indices = self.dictionary.sentence_to_indices(answer)

            X_train.append(q_indices)
            y_train.append(a_indices[:, 0])

        X_train = torch.cat(X_train, dim=0)
        y_train = torch.cat(y_train, dim=0)

        # Training setup
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self.model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        #save the trained model
        #torch.save(self.model.state_dict(), 'complete_ai_chatbot_model.pth')
        print("🎉 Training complete!")

    def chat(self, user_input):
        if self.model is None:
            return "I'm not ready yet! Please train me first."

        self.model.eval()

        # Analyze text complexity with NLTK
        complexity = self.knowledge_enhancer.analyze_text_complexity(user_input)

        # Advanced emotion detection with NLTK
        emotion, confidence = self.ei_system.advanced_emotion_detection(user_input)
        emotional_response = self.ei_system.get_empathetic_response(emotion, confidence, user_input)

        # Knowledge detection with NLTK keyword extraction
        knowledge_response = self.get_knowledge_response(user_input)

        # AI model response
        try:
            input_tensor = self.dictionary.sentence_to_indices(user_input)
            with torch.no_grad():
                output, _ = self.model(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()

            ai_word = self.dictionary.idx2word.get(predicted_idx, "<UNK>")

            if ai_word != "<UNK>":
                ai_response = ai_word.capitalize() + "!"
            else:
                ai_response = emotional_response

        except Exception as e:
            # Model inference failed; fallback to emotional response.
            print(f"Model inference error: {e}")
            ai_response = emotional_response

        # Combine responses with NLTK insights
        final_response = f"{emotional_response} {ai_response}"

        if knowledge_response:
            final_response += f"\n\n{knowledge_response}"

        # Add complexity note for long inputs
        if complexity == "complex":
            final_response += "\n\n💡 That was quite detailed! Thanks for the comprehensive input."

        # Remember conversation
        self.remember_conversation(user_input, final_response)

        return final_response

    def get_knowledge_response(self, user_input):
        input_lower = user_input.lower()

        if any(word in input_lower for word in ['what is', 'who is', 'explain', 'tell me about']):
            # Use NLTK for better topic extraction
            keywords = self.knowledge_enhancer.extract_keywords(user_input)
            if keywords:
                topic = " ".join(keywords[:2])
                #return f"🔍 Let me look up information about {topic}..."
                return self.knowledge_enhancer.wikipedia_lookup(topic) 
            

        elif 'time' in input_lower:
            now = datetime.datetime.now()
            return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

        elif any(word in input_lower for word in ['search', 'find', 'look up']):
            return self.knowledge_enhancer.web_search(user_input)

        return None

    def remember_conversation(self, user_input, response):
        conversation = {
            "user_input": user_input,
            "bot_response": response,
            "timestamp": datetime.datetime.now().isoformat()

        }
        self.conversation_memory.append(conversation)

    def show_nltk_capabilities(self):
        """Demonstrate NLTK features"""
        print("\n" + "="*50)
        print("🧠 NLTK CAPABILITIES DEMONSTRATION")
        print("="*50)

        test_text = "I am feeling incredibly happy and excited today because I learned about artificial intelligence!"

        # Tokenization
        tokens = word_tokenize(test_text)
        print(f"📝 Tokenization: {tokens[:5]}...")

        # Stopword removal
        filtered = [token for token in tokens if token not in self.dictionary.stop_words]
        print(f"🚫 Stopwords removed: {filtered[:5]}...")

        # Stemming
        stemmed = [self.dictionary.stemmer.stem(token) for token in filtered[:5]]
        print(f"✂️ Stemming: {stemmed}")

        # Sentiment analysis
        sentiment = self.ei_system.sia.polarity_scores(test_text)
        print(f"🎭 Sentiment Analysis: {sentiment}")

        print("✅ NLTK is fully integrated and working!")


    def interactive_chat(self):
        if self.model is None:
            print("❌ Chatbot not trained! Run setup() and train() first.")
            return

        print("\n" + "="*60)
        print("🤖 Hi, i am Upsilon C! 🚀")
        print("="*60)
        print("Type 'quit' to exit")
        print()

        while True:
            try:
                user_input = input("You: ").strip() #sl.listen2() #input("You: ").strip()


                if user_input.lower() == 'quit':
                    print("Bot: Goodbye! Thanks for chatting! 👋")

                    break
                elif user_input.lower() == 'train':
                    sl.speak("input an inter for epochs")
                    epoc=int(input())
                    if epoc < 20:
                        self.train(epochs=50)
                    else:
                        self.train(epochs=epoc)
                    continue
                elif user_input.lower() == 'memory':
                    print(f"Bot: I remember {len(self.conversation_memory)} conversations!")
                    #sl.speak("Bot: I remember {len(self.conversation_memory)} conversations!")
                    continue
                #sl.speak("i am here now at memory")
                response = self.chat(user_input)
                print(f"Bot: {response}")
                sl.speak(response)

            except KeyboardInterrupt:
                print("\nBot: Goodbye! 👋")
                sl.speak("Goodbye")
                break
            except Exception as e:
                print(f"Bot: Sorry, I had an error. Lets continue!")
                #sl.speak("Sorry, I had an error. Let's continue!")

            #sl.speak("if what is here is here i just dont know again")
            #response = self.chat(user_input)
            #print(f"Bot: {response}")
            #sl.speak(response)
