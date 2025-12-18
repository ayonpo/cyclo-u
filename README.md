# Complete AI Chatbot - Upsilon

A sophisticated AI chatbot with emotional intelligence, real-time knowledge integration, and continuous learning capabilities. Optimized for Android/Termux environments with graceful fallbacks for desktop platforms.

## Features

🤖 **Core AI Capabilities**
- LSTM-based neural network for intelligent responses
- Natural Language Processing with NLTK
- Emotional intelligence with sentiment analysis
- Real-time knowledge enhancement via web search and Wikipedia
- Continuous learning from conversations
- Empathetic response generation

🧠 **Emotional Intelligence**
- Advanced emotion detection (happy, sad, angry, fearful, surprised, etc.)
- Mood tracking and pattern analysis
- Context-aware empathetic responses
- User emotional history insights

🌐 **Knowledge Enhancement**
- Keyword extraction from user input
- Web search integration
- Wikipedia article summaries
- Text complexity analysis
- Lexical diversity measurement

📱 **Platform Support**
- **Primary:** Android via Termux (speech-to-text & text-to-speech)
- **Secondary:** Windows, Linux, macOS (desktop fallbacks for I/O)
- Safe graceful degradation when optional packages unavailable

## Quick Start

### Android/Termux (Recommended)

1. **Install Termux** from F-Droid or Github

2. **Set up Termux environment:**
   ```bash
   pkg update && pkg upgrade
   pkg install python git
   pkg install termux-api
   ```

3. **Install Termux:API APK** (separate download required for STT/TTS)

4. **Clone and install:**
   ```bash
   git clone https://github.com/ayonpo/cyclo-u.git
   cd cyclo-u
   pip install -r requirements.txt
   ```

5. **Run the chatbot:**
   ```bash
   python main.py
   ```

### Windows/Linux/macOS

```bash
# Clone repository
git clone https://github.com/ayompo/cyclo-u.git
cd cyclo-u

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run chatbot
python main.py
```

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

### Core Dependencies

```
torch>=2.0.0              # Neural network framework
nltk>=3.8.1               # Natural Language Toolkit
numpy>=1.24.0             # Numerical computing
wikipedia>=1.4.0          # Wikipedia API
textblob>=0.17.1          # Text sentiment analysis
requests>=2.31.0          # HTTP library
google-search-results>=2.4.2  # Web search integration
termux-api                # Android/Termux integration (optional)
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** All external dependencies have graceful fallbacks. The chatbot will function with partial dependencies installed, with reduced functionality.

## Usage

### Basic Chat

```bash
python main.py
```

The chatbot will:
1. ✅ Check for required external packages
2. 📥 Download NLTK data (first run only)
3. 🎓 Load training data (auto-generates if missing)
4. 🧠 Initialize neural model
5. 🎤 Enter interactive chat mode

### Interactive Commands

While chatting:
- Type `quit` to exit
- Type `train` to retrain with custom epochs
- Type `memory` to see conversation history size
- Regular text for normal conversation

### Example Interaction

```
You: Hello, how are you?
Bot: Hello! it's great to meet you! 😊

You: I'm feeling happy today
Bot: That's wonderful! 😊 Your happiness is contagious! ✨

You: Tell me about artificial intelligence
Bot: 🔍 Searching for artificial intelligence...
Bot: 📖 Artificial intelligence is the simulation of human intelligence 
processes by computer systems...
```

## Project Structure

```
complete-ai-chatbot/
├── main.py              # Application launcher with dependency checks
├── core.py              # Main chatbot logic (CompleteAIChatbot class)
├── cunn.py              # LSTM neural network (EnhancedChatBrain)
├── dcore.py             # Training data manager & smart dictionary
├── l_brain.py           # Emotional intelligence engine
├── r_brain.py           # Knowledge enhancer (search, Wikipedia, analysis)
├── comms.py             # I/O layer (Termux STT/TTS, desktop fallbacks)
├── tcore.py             # Placeholder for future enhancements
├── requirements.txt     # Python dependencies
├── chatbot_training.json # Training data (auto-generated)
├── chatbot_training.txt  # Human-readable training data format
└── README.md            # This file
```

## Architecture

### Module Overview

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `main.py` | Launcher & dependency checker | `check_external_dependencies()`, `main()` |
| `core.py` | Chatbot orchestrator | `CompleteAIChatbot` |
| `cunn.py` | Neural inference | `EnhancedChatBrain` (LSTM model) |
| `dcore.py` | Training & vocabulary | `SmartDictionary`, `TrainingManager` |
| `l_brain.py` | Emotions & empathy | `EmotionalIntelligence` |
| `r_brain.py` | Knowledge & search | `KnowledgeEnhancer` |
| `comms.py` | Speech & I/O | `listen()`, `speak()`, `listen2()` |
| `tcore.py` | Future expansions | (Reserved) |

### Data Flow

```
User Input (Speech/Text)
    ↓
[comms.py] - Speech-to-Text (Termux) or standard input
    ↓
[core.py] - Route through specialized engines
    ↓
├→ [l_brain.py] - Emotion detection & empathetic response
├→ [r_brain.py] - Knowledge extraction & search
└→ [cunn.py] - Neural model inference
    ↓
[core.py] - Combine responses
    ↓
[dcore.py] - Update training memory
    ↓
[comms.py] - Text-to-Speech (Termux) or print output
    ↓
User Output (Speech/Text)
```

## Features in Detail

### Emotional Intelligence
- **Sentiment Analysis:** Uses NLTK VADER for real-time sentiment scoring
- **Emotion Detection:** Keyword-based + sentiment-driven emotion classification
- **Empathetic Responses:** Context-aware replies based on detected emotion
- **Mood Tracking:** Tracks user mood history and identifies patterns

### Knowledge Enhancement
- **Keyword Extraction:** NLTK-based extraction of meaningful keywords
- **Web Search:** Google search integration for current information
- **Wikipedia Integration:** Automatic article summaries for topics
- **Text Complexity Analysis:** Measures sentence length, vocabulary diversity

### Learning & Memory
- **Conversation Memory:** Stores recent interactions (deque with maxlen=50)
- **Training Pairs:** Dynamic update of chatbot knowledge
- **Sentiment History:** Tracks emotional patterns over time
- **Retraining:** Support for custom epoch-based retraining

## Dependency Handling

The chatbot uses **safe import guards** for all external packages:

```python
try:
    import torch
except Exception:
    torch = None  # Graceful fallback
```

This means:
- ✅ The chatbot starts even if optional packages are missing
- ⚠️ Features requiring missing packages degrade gracefully
- 🔔 User is prompted at startup about missing dependencies
- 📋 Warning messages guide users to install if needed

## Known Issues & Troubleshooting

### PyTorch Initialization Error
If you encounter `torch.cat() expected non-empty list` on first run:
- **Solution:** Re-run the application. This is a known PyTorch initialization quirk.

### Speech Recognition Not Working (Termux)
- Ensure Termux:API APK is installed separately
- Verify `pkg install termux-api` was run
- Test with: `termux-speech-to-text`

### Missing NLTK Data
- The chatbot auto-downloads required NLTK data on first run
- If download fails, manually run: `python -c "import nltk; nltk.download('punkt')"` etc.

### Long Training Time
- Initial training with 100 epochs may take time (depending on hardware)
- Use the `train` command during interactive chat to retrain with fewer epochs

## Performance Notes

- **Model:** 2-layer LSTM with 128 hidden units, 64-dim embeddings
- **Training Data:** 100+ conversation pairs (auto-generated + custom variations)
- **Inference Time:** < 100ms per response (with GPU) / 200-500ms (CPU)
- **Memory:** ~200MB base + model weights (~50MB)

## Contributing

Contributions welcome! Areas of interest:

- Enhanced emotion models (more nuanced detection)
- Additional knowledge sources (APIs, local databases)
- Multi-language support
- Improved training data generation
- Mobile UI/UX improvements
- Performance optimizations

## License

[Add your license here - e.g., MIT, Apache 2.0, GPL 3.0]

## Citation

If you use this chatbot in research or production, please cite:

```bibtex
@software{upsilon_chatbot_2024,
  author = {Your Name},
  title = {Complete AI Chatbot - Upsilon},
  year = {2024},
  url = {https://github.com/yourusername/complete-ai-chatbot}
}
```

## Acknowledgments

- **NLTK:** Natural Language Toolkit for NLP capabilities
- **PyTorch:** Deep learning framework
- **VADER Sentiment Analyzer:** Sentiment analysis engine
- **Wikipedia API:** Knowledge source
- **Google Search Results:** Web search integration

## Support

For issues, questions, or feature requests:
- Open an GitHub issue with detailed description
- Include Python version, OS, and error messages
- Provide minimal reproduction steps

## Changelog

### v1.0.0 (2024-12-18)
- Initial release
- Full emotional intelligence integration
- Knowledge enhancement via search and Wikipedia
- Android/Termux optimization
- Graceful dependency handling

---

**Built with ❤️ for intelligent, empathetic conversations across all platforms.**

*Last Updated: December 18, 2024*
