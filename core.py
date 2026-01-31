"""Complete AI chatbot with features integration."""

import datetime
import json
import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F 
except Exception:
    torch = None
    nn = None
    optim = None

try:
    from nltk.tokenize import word_tokenize
except Exception:
    word_tokenize = None

# First-party/local imports
from c_nn import EnhancedChatBrain
from dcore import TrainingManager, SmartDictionary
from l_brain import EmotionalIntelligence
from r_brain import KnowledgeEnhancer

import comms as sl

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
        print("🔄 Setting up NLTK...")

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


    def train(self, epochs=180, learning_rate=0.01):
        if self.model is None:
            return

        print("🎯 Training AI...")
        sl.speak("training commencing")

        # Prepare training data

        X_train = []
        y_train = []

        for question, answer in self.trainer.training_pairs:
            q_indices = self.dictionary.sentence_to_indices(question)
            a_indices = self.dictionary.sentence_to_indices(answer)

            X_train.append(q_indices)
            y_train.append(a_indices)

        X_train = torch.cat(X_train, dim=0)
        y_train = torch.cat(y_train, dim=0)

        # Training setup
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <PAD>
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self.model(X_train)
            loss = criterion(output.view(-1, self.dictionary.vocab_size), y_train.view(-1))
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        #save the trained model
        #torch.save(self.model.state_dict(), 'complete_ai_chatbot_model.pth')
        print("🎉 Training complete!")

    def chat(self, user_input):
        if self.model is None:
            return "not ready yet! activate model training first."

        self.model.eval()

        # Analyze text complexity with NLTK
        complexity = self.knowledge_enhancer.analyze_text_complexity(user_input)

        # Advanced emotion detection with NLTK
        emotion, confidence = self.ei_system.advanced_emotion_detection(user_input)
        emotional_response = self.ei_system.get_empathetic_response(emotion, confidence, user_input)

        # Knowledge detection with NLTK keyword extraction
        knowledge_response = self.get_knowledge_response(user_input)
        """
        method for response:version one 1
        """
        input_tensor = self.dictionary.sentence_to_indices(user_input)
        #new input starts here

        with torch.no_grad():

            output, _ = self.model(input_tensor)
            predicted_indices = torch.argmax(output, dim=-1)[0]  # (seq_len,)
            
            words = []
            for idx in predicted_indices:
                idx = idx.item()
                if idx == 0:  # <PAD>
                    break
                word = self.dictionary.idx2word.get(idx, "")
                if word and word != "<UNK>":
                    words.append(word)
            
            ai_response = " ".join(words).capitalize() + "!" if words else "I don't know what to say."
        
        # Combine responses with NLTK insights
        final_response = f"{emotional_response} {ai_response}"

        if knowledge_response:
            final_response += f"\n\n{knowledge_response}"

        # Add complexity note for long inputs
        if complexity == "complex":
            final_response += "\n\n💡 That was quite detailed! Thanks for the comprehensive input."

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

        elif any(word in input_lower for word in ['learn about', 'research', 'study']):
            return self.knowledge_enhancer.search_and_learn(user_input)

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
        #print(f"📝 Tokenization: {tokens[:5]}...")

        # Stopword removal
        filtered = [token for token in tokens if token not in self.dictionary.stop_words]
        #print(f"🚫 Stopwords removed: {filtered[:5]}...")

        # Stemming
        stemmed = [self.dictionary.stemmer.stem(token) for token in filtered[:5]]
        #print(f"✂️ Stemming: {stemmed}")

        # Sentiment analysis
        sentiment = self.ei_system.sia.polarity_scores(test_text)
        #print(f"🎭 Sentiment Analysis: {sentiment}")

        #print("✅ NLTK is fully integrated and working!")


    def interactive_chat(self):
        if self.model is None:
            print("❌ Cyclo not trained! Run setup() and train() first.")
            return

        print("\n" + "="*60)
        print("🤖 Hi, i am Cyclo Bond! 🚀")
        print("="*60)
        print("Type 'quit' to exit")
        print()
        while True:
            try:
                user_input = input("You: ").strip() #sl.listen2() #input("You: ").strip()

                if user_input.lower() == 'quit':
                    print("Bot: Goodbye! Thanks for chatting! 👋")
                    sl.speak('hope you had a good time!')
                    break
                elif user_input.lower() == 'train':
                    sl.speak("input an integer for epochs")
                    try:
                        epochs = int(input("Number of epochs: "))
                        if epochs <= 0:
                            print("❌ Epochs must be positive!")
                            sl.speak("Epochs must be positive")
                        elif epochs > 500:
                            print("⚠️ That's a lot of epochs! Using 200 instead.")
                            sl.speak("Using 200 epochs instead")
                            self.train(epochs=200)
                        else:
                            self.train(epochs=epochs)
                    except ValueError:
                        print("❌ Please enter a valid number!")
                        sl.speak("Please enter a valid number")
                    continue
                elif user_input.lower() in ['help', 'commands', '?']:
                    self.show_help()
                    continue
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                elif user_input.lower() == 'add data':
                    # Bulk add data
                    print("\n📥 BULK DATA IMPORT")
                    print("1. Import from JSON")
                    print("2. Import from CSV")
                    print("3. Import from text file")
                    print("4 or other number. continue and exit data add mode")
            
                    choice = input("Choice: ").strip()
            
                    if choice == '1':
                        filename = input("JSON filename: ").strip() or "additional_training.json"
                        self.trainer.load_training_data(filename)
                    elif choice == '2':
                        filename = input("CSV filename: ").strip() or "training_data.csv"
                        self.trainer.import_from_csv(filename)
                    elif choice == '3':
                        filename = input("Text filename: ").strip() or "conversations.txt"
                        self.trainer.import_from_text_file(filename)
                    else:
                        continue
                                  
                elif user_input.lower() == 'web learn':
                    topic = input("What topic should I learn about from the web? ").strip()
                    if topic:
                        result = self.knowledge_enhancer.search_and_learn(topic)
                        print(f"Bot: {result}")
                        sl.speak("Web learning completed")
                    continue
                elif user_input.lower() == 'chat':
                    while True:
                        sl.speak("you can type exit to go back, learning mode to toggle learning, or just chat normally. Learning mode is currently " + ("enabled" if self.learning_mode else "disabled"))
                        chat_input = input("You (type 'exit' to go back, 'learning mode' to toggle): ").strip()
                        if chat_input.lower() == 'back' or chat_input.lower() == 'exit':
                            break
                        elif chat_input.lower() == 'learning mode':
                            self.learning_mode = not self.learning_mode
                            status = "ENABLED" if self.learning_mode else "DISABLED"
                            print(f"🧠 Learning mode {status}")
                            sl.speak(f"Learning mode {status.lower()}")
                            continue
                        else:
                            if self.learning_mode:
                                response = self.chat_and_learn(chat_input)
                            else:
                                response = self.chat(chat_input)
                            print(f"Bot: {response}")
                            sl.speak(response)
                    #response = self.chat(user_input)
                    #print(f"Bot: {response}")
                    #sl.speak(response)
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    sl.speak("Unknown command. Type help for available commands")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye! 👋")
                sl.speak("Goodbye")
                break
            except Exception as e:
                print(f"Bot: Sorry, I had an error. Lets continue!")
                sl.speak("Sorry, I had an error. Let's continue!")


    def show_help(self):
        """Display available commands"""
        print("\n" + "="*60)
        print("🤖 CYCLO-U CHATBOT COMMANDS")
        print("="*60)
        print("📋 MAIN COMMANDS:")
        print("  chat         - Start interactive chat mode")
        print("  train        - Train the model with custom epochs")
        print("  learning mode- Toggle automatic learning on/off")
        print("  status       - Show current bot status")
        print("  add data     - Import training data from files")
        print("  web learn    - Learn from web search results")
        print("  help         - Show this help message")
        print("  quit         - Exit the chatbot")
        print()
        print("🎯 LEARNING FEATURES:")
        print(f"  Learning mode: {'ENABLED' if self.learning_mode else 'DISABLED'}")
        print("  - When enabled, bot learns from all conversations")
        print("  - Retrains automatically every 10 new examples")
        print()
        print("💾 DATA MANAGEMENT:")
        print("  - Supports JSON, CSV, and text file imports")
        print("  - Automatic conversation memory (last 50 chats)")
        print("  - Incremental learning with vocabulary expansion")
        print("="*60)

        sl.speak("Here are the available commands. Check the console for details.")


    def show_status(self):
        """Display current bot status"""
        print("\n" + "="*60)
        print("📊 CYCLO-U BOT STATUS")
        print("="*60)
        print(f"🧠 Learning Mode: {'ENABLED' if self.learning_mode else 'DISABLED'}")
        print(f"🗣️ Conversations in Memory: {len(self.conversation_memory)}")
        print(f"📚 Training Pairs Loaded: {len(self.trainer.training_pairs) if hasattr(self.trainer, 'training_pairs') else 'Unknown'}")
        print(f"🔤 Dictionary Size: {self.dictionary.vocab_size if hasattr(self.dictionary, 'vocab_size') else 'Unknown'} words")
        print(f"🤖 Model Status: {'Loaded' if self.model is not None else 'Not Loaded'}")
        if hasattr(self, 'incremental_trainer') and self.incremental_trainer.new_data_buffer:
            print(f"📝 Pending Training Examples: {len(self.incremental_trainer.new_data_buffer)}")
        print("="*60)

        status_msg = f"Learning mode is {'enabled' if self.learning_mode else 'disabled'}. "
        status_msg += f"I remember {len(self.conversation_memory)} conversations."
        sl.speak(status_msg)


class IncrementalTrainer:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.new_data_buffer = []
        self.retrain_every = 10  # Retrain after 10 new examples
        self.training_history = []
    
    def add_conversation(self, user_input, bot_response, confidence=1.0):
        """Add a conversation to training buffer"""
        if confidence > 0.7:  # Only add high-confidence responses
            self.new_data_buffer.append((user_input, bot_response))
            print(f"📝 Added to training buffer: '{user_input}' -> '{bot_response}'")
            
            # Check if we should retrain
            if len(self.new_data_buffer) >= self.retrain_every:
                self.retrain_with_new_data()
    
    def retrain_with_new_data(self):
        """Retrain model with accumulated new data"""
        if not self.new_data_buffer:
            return
        
        print(f"\n🔄 Retraining with {len(self.new_data_buffer)} new examples...")
        
        # Add to main training data
        self.chatbot.trainer.training_pairs.extend(self.new_data_buffer)
        
        # Update dictionary
        for question, answer in self.new_data_buffer:
            self.chatbot.dictionary.add_sentence(question)
            self.chatbot.dictionary.add_sentence(answer)
        
        # Rebuild model with new vocabulary size
        old_vocab_size = self.chatbot.model.fc.out_features
        new_vocab_size = self.chatbot.dictionary.vocab_size
        
        if new_vocab_size > old_vocab_size:
            print(f"📈 Expanding vocabulary from {old_vocab_size} to {new_vocab_size} words")
            self.expand_model_vocabulary(new_vocab_size)
        
        # Continue training
        self.continue_training(epochs=50)
        
        # Clear buffer
        self.new_data_buffer.clear()
        print("✅ Retraining complete!")
    
    def expand_model_vocabulary(self, new_vocab_size):
        """Expand model to handle new vocabulary"""
        # Get current model parameters
        old_fc = self.chatbot.model.fc
        
        # Create new fully connected layer with expanded output
        new_fc = nn.Linear(old_fc.in_features, new_vocab_size)
        
        # Copy old weights (for existing vocabulary)
        with torch.no_grad():
            new_fc.weight[:old_fc.out_features] = old_fc.weight
            new_fc.bias[:old_fc.out_features] = old_fc.bias
            
            # Initialize new weights randomly
            nn.init.xavier_uniform_(new_fc.weight[old_fc.out_features:])
            nn.init.zeros_(new_fc.bias[old_fc.out_features:])
        
        # Replace the layer
        self.chatbot.model.fc = new_fc
        print(f"🔧 Model expanded to handle {new_vocab_size} words")
    
    def continue_training(self, epochs=50, learning_rate=0.001):
        """Continue training from where we left off"""
        print(f"🎯 Continuing training for {epochs} epochs...")
        
        # Prepare all training data
        X_train = []
        y_train = []
        
        for question, answer in self.chatbot.trainer.training_pairs:
            q_indices = self.chatbot.dictionary.sentence_to_indices(question)
            a_indices = self.chatbot.dictionary.sentence_to_indices(answer)
            
            X_train.append(q_indices)
            y_train.append(a_indices)
        
        X_train = torch.cat(X_train, dim=0)
        y_train = torch.cat(y_train, dim=0)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.chatbot.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.chatbot.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self.chatbot.model(X_train)
            loss = criterion(output.view(-1, self.chatbot.dictionary.vocab_size), y_train.view(-1))
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'  Epoch {epoch}, Loss: {loss.item():.4f}')
        
        self.training_history.append({
            'epochs': epochs,
            'new_samples': len(self.new_data_buffer),
            'vocab_size': self.chatbot.dictionary.vocab_size,
            'timestamp': datetime.datetime.now().isoformat()
        })

class AutoLearningChatbot(CompleteAIChatbot):
    def __init__(self):
        super().__init__()
        self.incremental_trainer = IncrementalTrainer(self)
        self.learning_log = []
        self.learning_mode = False  # Toggle for continuous learning
        
    def chat_and_learn(self, user_input):
        """Chat and automatically learn from good conversations"""
        # Get response
        response = self.chat(user_input)

        # Learn if the response seems reasonable (basic heuristics)
        if self.should_learn_from_conversation(user_input, response):
            self.incremental_trainer.add_conversation(user_input, response, confidence=0.8)
            print("🧠 Auto-learned from this conversation!")

        # Occasionally ask for feedback (reduced frequency)
        if random.random() < 0.05:  # 5% chance to ask for feedback
            print("\n🤔 Was that response helpful? (yes/no/correct me)")
            try:
                feedback = input("Your feedback (or press Enter to skip): ").strip().lower()
                if feedback.startswith('correct'):
                    print("📝 Please provide the correct response:")
                    corrected = input("Correct response: ").strip()
                    self.learn_from_correction(user_input, corrected)
                elif 'yes' in feedback:
                    self.incremental_trainer.add_conversation(user_input, response, confidence=1.0)
                    print("✅ Confirmed as good response!")
            except:
                pass  # Skip feedback if input fails

        return response

    def should_learn_from_conversation(self, user_input, response):
        """Determine if this conversation is worth learning from"""
        # Basic heuristics for learning
        if len(user_input.split()) < 2 or len(response.split()) < 1:
            return False  # Too short

        if any(word in user_input.lower() for word in ['quit', 'exit', 'help', 'train']):
            return False  # Command-like inputs

        if response.lower() in ['i don\'t know', 'sorry', 'error']:
            return False  # Poor responses

        return True  # Worth learning
    
    def learn_from_success(self, user_input, response, confidence=0.8):
        """Learn from successful responses"""
        # More lenient learning criteria
        if len(user_input.split()) >= 1 and len(response.split()) >= 1:
            self.incremental_trainer.add_conversation(user_input, response, confidence)
            self.learning_log.append({
                'type': 'auto_learn',
                'input': user_input,
                'response': response,
                'confidence': confidence,
                'timestamp': datetime.datetime.now().isoformat()
            })
    
    def learn_from_correction(self, user_input, corrected_response):
        """Learn from user corrections"""
        self.incremental_trainer.add_conversation(user_input, corrected_response)
        self.learning_log.append({
            'type': 'user_correction',
            'input': user_input,
            'corrected_response': corrected_response,
            'timestamp': datetime.datetime.now().isoformat()
        })
        print("✅ Learned from your correction!")
    
    def export_learning_log(self, filename="learning_log.json"):
        """Export what the bot has learned"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.learning_log, f, indent=2, ensure_ascii=False)
        print(f"💾 Learning log exported to {filename}")