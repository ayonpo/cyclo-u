
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
    import re
except Exception:
    re = None

try:
    from googlesearch import search
except Exception:
    search = None

try:
    import wikipedia
except Exception:
    wikipedia = None

try:
    import requests
    from bs4 import BeautifulSoup
    import urllib.parse
    import time
    import random
except Exception:
    requests = None
    BeautifulSoup = None

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
        self.content_cache = {}
        self.last_request_time = 0
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]

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

    def web_search(self, query, max_results=3, include_content=True):
        """Enhanced search with content extraction and summarization"""
        try:
            # Rate limiting - wait at least 1 second between requests
            current_time = time.time()
            if current_time - self.last_request_time < 1:
                time.sleep(1 - (current_time - self.last_request_time))
            self.last_request_time = time.time()

            keywords = self.extract_keywords(query)
            print(f"🔍 Keywords extracted: {keywords}")

            if not keywords:
                return "I couldn't identify clear keywords to search for."

            # Use keywords for more focused search
            search_query = " ".join(keywords[:4])  # Use top 4 keywords
            print(f"🔍 Searching for: {search_query}")

            if search is None:
                return "Search functionality not available (missing package)."

            # Check cache first
            if search_query in self.search_cache:
                print("📋 Using cached search results")
                urls = self.search_cache[search_query]
            else:
                try:
                    urls = list(search(search_query, num_results=max_results, lang='en'))
                    self.search_cache[search_query] = urls
                except Exception as e:
                    return f"Search failed: {e}"

            if not urls:
                return f"I couldn't find recent information about '{search_query}'."

            # If content extraction is requested
            if include_content and requests and BeautifulSoup:
                return self.extract_and_summarize_content(urls, search_query)
            else:
                # Fallback to just URLs
                result = f"I found information about '{search_query}':\n"
                for i, url in enumerate(urls[:max_results], 1):
                    result += f"{i}. {url}\n"
                return result

        except Exception as e:
            return f"Search unavailable right now. Error: {e}"

    def extract_and_summarize_content(self, urls, query):
        """Extract content from URLs and provide summaries"""
        summaries = []
        successful_extractions = 0

        for url in urls[:3]:  # Limit to first 3 URLs
            try:
                content = self.scrape_webpage_content(url)
                if content:
                    summary = self.summarize_content(content, query)
                    if summary:
                        summaries.append(f"📄 From {url.split('//')[-1].split('/')[0]}: {summary}")
                        successful_extractions += 1

                # Rate limiting between page requests
                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"⚠️ Failed to extract from {url}: {e}")
                continue

        if summaries:
            result = f"🔍 Search results for '{query}':\n\n"
            result += "\n\n".join(summaries)
            if successful_extractions < len(urls):
                result += f"\n\n📊 Successfully extracted content from {successful_extractions}/{len(urls)} sources."
            return result
        else:
            # Fallback to URLs only
            result = f"I found information about '{query}' but couldn't extract content:\n"
            for i, url in enumerate(urls, 1):
                result += f"{i}. {url}\n"
            return result

    def scrape_webpage_content(self, url):
        """Safely scrape webpage content with proper headers"""
        if not requests or not BeautifulSoup:
            return None

        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text from main content areas
            content_selectors = [
                'main', 'article', '.content', '#content',
                '.post-content', '.entry-content', 'body'
            ]

            text_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    text_content = elements[0].get_text(separator=' ', strip=True)
                    break

            if not text_content:
                # Fallback to body text
                text_content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""

            # Clean up the text
            text_content = self.clean_extracted_text(text_content)

            return text_content[:2000] if len(text_content) > 2000 else text_content

        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
            return None

    def clean_extracted_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove common unwanted phrases
        unwanted_patterns = [
            r'Cookie Policy', r'Privacy Policy', r'Terms of Service',
            r'© \d{4}', r'All rights reserved', r'Follow us on',
            r'Share this', r'Related Articles', r'Advertisement'
        ]

        for pattern in unwanted_patterns:
            text = text.replace(pattern, '')

        return text.strip()

    def summarize_content(self, content, query, max_sentences=2):
        """Create a simple summary of the content"""
        if not content or not sent_tokenize:
            return content[:200] + "..." if len(content) > 200 else content

        try:
            sentences = sent_tokenize(content)

            # Simple relevance scoring based on keyword matches
            keywords = set(self.extract_keywords(query))
            sentence_scores = []

            for sentence in sentences:
                sentence_words = set(self.extract_keywords(sentence))
                score = len(keywords.intersection(sentence_words))
                sentence_scores.append((sentence, score))

            # Sort by relevance and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in sentence_scores[:max_sentences]]

            # Return in original order if possible
            summary = ' '.join(top_sentences[:max_sentences])

            return summary if summary else content[:300] + "..."

        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
            return content[:300] + "..." if len(content) > 300 else content
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
    
    def scrape_conversation_data(self, url=None, max_pairs=50):
        """Scrape conversation data from websites with improved parsing"""
        if not requests or not BeautifulSoup:
            return []

        if url is None:
            # Default to a conversation dataset source
            url = "https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html"

        try:
            print(f"🌐 Scraping conversation data from: {url}")

            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            new_pairs = []

            # Try multiple strategies for extracting conversations
            strategies = [
                self._extract_from_dialogue_tags,
                self._extract_from_qa_pairs,
                self._extract_from_structured_data,
                self._extract_from_text_blocks
            ]

            for strategy in strategies:
                pairs = strategy(soup)
                if pairs:
                    new_pairs.extend(pairs)
                    if len(new_pairs) >= max_pairs:
                        break

            # Clean and filter pairs
            filtered_pairs = []
            for question, answer in new_pairs[:max_pairs]:
                if self._is_valid_conversation_pair(question, answer):
                    filtered_pairs.append((question.strip(), answer.strip()))

            print(f"✅ Scraped {len(filtered_pairs)} valid conversation pairs")
            return filtered_pairs

        except Exception as e:
            print(f"❌ Web scraping failed: {e}")
            return []

    def _extract_from_dialogue_tags(self, soup):
        """Extract from dialogue-specific HTML tags"""
        pairs = []
        dialogues = soup.find_all(['div', 'p', 'span'], class_=re.compile(r'dialogue|conversation|chat', re.I))

        for dialogue in dialogues:
            text = dialogue.get_text(strip=True)
            # Split on common dialogue markers
            parts = re.split(r'[:\n-]', text, 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))

        return pairs

    def _extract_from_qa_pairs(self, soup):
        """Extract question-answer pairs"""
        pairs = []

        # Look for Q&A patterns
        questions = soup.find_all(['div', 'p', 'li'], string=re.compile(r'^(Q|Question):', re.I))
        answers = soup.find_all(['div', 'p', 'li'], string=re.compile(r'^(A|Answer):', re.I))

        for q, a in zip(questions, answers):
            q_text = re.sub(r'^(Q|Question):\s*', '', q.get_text(), flags=re.I)
            a_text = re.sub(r'^(A|Answer):\s*', '', a.get_text(), flags=re.I)
            pairs.append((q_text.strip(), a_text.strip()))

        return pairs

    def _extract_from_structured_data(self, soup):
        """Extract from structured data like tables or lists"""
        pairs = []

        # Try tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    pairs.append((cells[0].get_text().strip(), cells[1].get_text().strip()))

        return pairs

    def _extract_from_text_blocks(self, soup):
        """Extract from general text blocks using heuristics"""
        pairs = []

        # Get all paragraphs
        paragraphs = soup.find_all('p')
        text_blocks = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]

        # Look for conversational patterns
        for i in range(len(text_blocks) - 1):
            current = text_blocks[i]
            next_block = text_blocks[i + 1]

            # Simple heuristic: if current ends with question mark and next is reasonable length
            if current.endswith('?') and 10 < len(next_block) < 200:
                pairs.append((current, next_block))

        return pairs

    def _is_valid_conversation_pair(self, question, answer):
        """Validate if a question-answer pair is suitable for training"""
        if not question or not answer:
            return False

        # Check lengths
        if len(question) < 3 or len(answer) < 3:
            return False

        if len(question) > 200 or len(answer) > 200:
            return False

        # Check for unwanted content
        unwanted_words = ['http', 'www.', '.com', 'javascript:', 'function(']
        combined_text = (question + answer).lower()

        if any(word in combined_text for word in unwanted_words):
            return False

        return True

    def search_and_learn(self, query, max_pages=2):
        """Search web and extract training data from results"""
        if not requests or not BeautifulSoup:
            return "Web scraping not available (missing requests/beautifulsoup4)"

        try:
            print(f"🔍 Searching and learning about: {query}")

            # Get search results
            if search is None:
                return "Search functionality not available"

            urls = list(search(query, num_results=max_pages, lang='en'))

            training_pairs = []
            extracted_info = []

            for url in urls:
                try:
                    content = self.scrape_webpage_content(url)
                    if content:
                        # Extract potential Q&A pairs from content
                        pairs = self.extract_qa_from_content(content, query)
                        training_pairs.extend(pairs)

                        # Also extract general information
                        summary = self.summarize_content(content, query, max_sentences=1)
                        if summary:
                            extracted_info.append(f"From {url.split('//')[-1].split('/')[0]}: {summary}")

                    time.sleep(random.uniform(1, 2))  # Rate limiting

                except Exception as e:
                    print(f"⚠️ Failed to process {url}: {e}")
                    continue

            if training_pairs:
                print(f"📚 Extracted {len(training_pairs)} training pairs from web search")

            result = f"🔍 Web Learning Results for '{query}':\n\n"
            if extracted_info:
                result += "📄 Key Information:\n" + "\n".join(f"• {info}" for info in extracted_info[:3])

            if training_pairs:
                result += f"\n\n🤖 Learned {len(training_pairs)} conversation patterns"

            return result

        except Exception as e:
            return f"Web learning failed: {e}"

    def extract_qa_from_content(self, content, topic):
        """Extract question-answer pairs from webpage content"""
        pairs = []

        if not sent_tokenize:
            return pairs

        try:
            sentences = sent_tokenize(content)

            # Look for patterns that might indicate Q&A
            for i in range(len(sentences) - 1):
                current = sentences[i].strip()
                next_sent = sentences[i + 1].strip()

                # Question followed by answer pattern
                if (current.endswith('?') and len(current) < 100 and
                    len(next_sent) > 10 and len(next_sent) < 300):
                    pairs.append((current, next_sent))

                # Topic explanation pattern
                topic_words = set(self.extract_keywords(topic))
                if (len(current) < 150 and len(next_sent) > 50 and
                    any(word.lower() in current.lower() for word in topic_words)):
                    pairs.append((f"What is {current.lower()}?", next_sent))

        except Exception as e:
            print(f"⚠️ Q&A extraction failed: {e}")

        return pairs[:10]  # Limit to 10 pairs per page

    def get_webpage_metadata(self, url):
        """Extract metadata from webpage"""
        if not requests or not BeautifulSoup:
            return {}

        try:
            headers = {'User-Agent': random.choice(self.user_agents)}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            metadata = {
                'title': soup.title.string if soup.title else '',
                'description': '',
                'keywords': '',
                'author': '',
                'publish_date': ''
            }

            # Extract meta tags
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name', '').lower()
                content = tag.get('content', '')

                if name == 'description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
                elif name == 'author':
                    metadata['author'] = content
                elif name in ['article:published_time', 'publishdate']:
                    metadata['publish_date'] = content

            return metadata

        except Exception as e:
            print(f"⚠️ Failed to extract metadata from {url}: {e}")
            return {}