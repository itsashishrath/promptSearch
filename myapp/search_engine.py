from rank_bm25 import BM25Okapi
import json
import numpy as np
from pathlib import Path
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict, Any

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResumeSearchEngine:
    def __init__(self, dataset_path: str = "resume_dataset.json"):
        """
        Initialize the search engine with the resume dataset.

        Args:
            dataset_path (str): Path to the JSON dataset containing resume texts
        """
        self.dataset_path = Path(dataset_path)
        self.documents = []
        self.bm25 = None
        self.stop_words = set(stopwords.words('english'))

        # Load and initialize the search engine
        self.load_dataset()
        self.initialize_bm25()

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing and removing stopwords.

        Args:
            text (str): Input text

        Returns:
            List[str]: List of preprocessed tokens
        """
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens
                 if token not in self.stop_words and token.isalpha()]

        return tokens

    def load_dataset(self):
        """Load and preprocess the resume dataset."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

            logger.info(f"Loaded {len(self.documents)} documents from dataset")

            # Preprocess all documents
            for doc in self.documents:
                doc['tokenized_text'] = self.preprocess_text(doc['text'])

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def initialize_bm25(self):
        """Initialize the BM25 model with the preprocessed documents."""
        try:
            # Create corpus for BM25
            corpus = [doc['tokenized_text'] for doc in self.documents]

            # Initialize BM25
            self.bm25 = BM25Okapi(corpus)

            logger.info("BM25 model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing BM25: {str(e)}")
            raise

    def search(self, prompt: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for resumes matching the given prompt.

        Args:
            prompt (str): Search prompt
            top_k (int): Number of top results to return

        Returns:
            List[Dict]: List of top matching documents with scores
        """
        try:
            # Preprocess the query
            tokenized_prompt = self.preprocess_text(prompt)

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_prompt)

            # Get top k document indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Prepare results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    doc = self.documents[idx]
                    result = {
                        'id': doc['id'],
                        'filename': doc['filename'],
                        'score': float(scores[idx]),  # Convert numpy float to Python float
                        'preview': ' '.join(doc['text'].split()[:50]) + '...'  # First 50 words preview
                    }
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise