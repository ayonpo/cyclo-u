"""Main launcher for the Complete AI Chatbot with all integrated features."""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

try:
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

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

from core import AutoLearningChatbot
import comms as sl
import dcore as dc
# Dependency checker for external packages (non-blocking by default)
import importlib
import importlib.util


def check_file_imports():
    """Check that all project files can be imported successfully"""
    print("🔍 Checking file imports and integrity...")
    
    files_to_check = [
        ('core', 'CompleteAIChatbot, AutoLearningChatbot'),
        ('cunn', 'EnhancedChatBrain'),
        ('dcore', 'TrainingManager, SmartDictionary'),
        ('l_brain', 'EmotionalIntelligence'),
        ('r_brain', 'KnowledgeEnhancer'),
        ('comms', 'speak, listen'),
        ('tcore', 'IncrementalTrainer')
    ]
    
    all_good = True
    
    for module_name, expected_classes in files_to_check:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}.py imported successfully")
            
            # Check if expected classes/functions exist
            for item in expected_classes.split(', '):
                if hasattr(module, item.strip()):
                    print(f"   ✓ {item} found")
                else:
                    print(f"   ❌ {item} NOT found in {module_name}")
                    all_good = False
                    
        except ImportError as e:
            print(f"❌ Failed to import {module_name}.py: {e}")
            all_good = False
        except Exception as e:
            print(f"⚠️  {module_name}.py imported but has issues: {e}")
            all_good = False
    
    return all_good


def check_external_dependencies(mapping=None, abort_on_missing=False):
    """Check that external modules are importable.

    mapping: dict[str, list[str]] or dict[str,str]
      keys are package labels, values are import names or list of possible import names.
    Returns a list of missing package labels. If abort_on_missing is True and any
    required packages are missing, the function will raise ImportError.
    """
    if mapping is None:
        mapping = {
            "torch": ["torch"],
            "nltk": ["nltk"],
            "wikipedia": ["wikipedia"],
            "textblob": ["textblob"],
            "requests": ["requests"],
            "googlesearch": ["googlesearch"],
            "termux-api": ["termux_api", "termux"]
        }

    missing = []
    available = []
    for label, imports in mapping.items():
        if isinstance(imports, str):
            imports = [imports]
        found = False
        for mod in imports:
            try:
                if importlib.util.find_spec(mod) is not None:
                    found = True
                    break
            except Exception:
                # treat as not found
                pass
        if not found:
            missing.append(label)
        else:
            available.append(label)

    if missing:
        print("[dependency-check] Missing external packages:", ", ".join(missing))
        print("[dependency-check] Installed packages:", ", ".join(available))
        if abort_on_missing:
            raise ImportError(f"Missing required packages: {', '.join(missing)}")
        # interactive prompt: allow user to continue or abort
        try:
            ans = input("Some optional packages are missing. Continue anyway? [y/N]: ")
        except Exception:
            ans = "n"
        if ans.strip().lower() not in ("y", "yes"):
            print("Aborting start due to missing dependencies.")
            return missing, True

    return missing, False


# ========================
# 🚀 MAIN LAUNCHER
# ========================


def root():
    print("Initiating Upsilon boot up sequence")
    print("This may take a moment...\n")
    
    # Check file integrity first
    print("📁 Checking project file integrity...")
    if not check_file_imports():
        print("❌ File integrity check failed! Cannot proceed.")
        sl.speak('file integrity check failed')
        return
    print("✅ All project files validated successfully!\n")
    
    # Lightweight dependency check using built-in defaults (no requirements parsing)
    """"
    try:
        missing, aborted = check_external_dependencies(mapping=None, abort_on_missing=False)
    except ImportError as exc:
        print(f"Dependency check failed: {exc}")
        return

    if aborted:
        # user chose to abort during dependency prompt
        return

    if missing:
        print("[dependency-check] Warning: missing optional packages - continuing with degraded functionality.")
        try:
            sl.speak('Warning: some optional packages are missing. Functionality may be reduced.')
        except Exception:
            pass
            
            """
    dc.download_nltk_data()
    #print('dowload completed and back to main file')
    sl.speak('booting up, wait for my initialisation')

    # instantiate chatbot
    cyclobot = AutoLearningChatbot()
    print('instantiation done')

    # Setup (load data, build dictionary, initialize model)
    
    if not cyclobot.setup():
        print(" Failed to setup AI!")
        sl.speak('my set up is missing')
        return
    # Show NLTK capabilities
    #cyclobot.show_nltk_capabilities()

    #Train the model
    print("\n⏳ Training AI...")
    sl.speak('data training commencing')
    cyclobot.train(epochs=200, learning_rate=0.01)

    # Start chatting!

    sl.speak('Training is complete')
    cyclobot.interactive_chat()

if __name__ == "__main__":
    #testing the speach
    print('Cyclo upsilon setup starting')
    root()