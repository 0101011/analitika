import json
import itertools
import ssl
import warnings
from os import path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
import h5py

from config import get_config

CONFIG = get_config()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' resource not found. Attempting to download...")
        try:
            nltk.download('punkt', quiet=True)
        except ssl.SSLError:
            warnings.warn(
                "SSL certificate verification failed. Attempting to download NLTK data without verification. "
                "This is not secure and should only be used for testing purposes.",
                UserWarning
            )
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download('punkt', quiet=True)

# Download necessary NLTK data
download_nltk_data()

WHITELIST = CONFIG['WHITELIST']
VOCAB_SIZE = CONFIG['VOCAB_SIZE']
UNK = 'unk'

WHITELIST = CONFIG['WHITELIST']
VOCAB_SIZE = CONFIG['VOCAB_SIZE']
UNK = 'unk'

limit = {
    'max_descriptions': CONFIG['MAX_DESCRIPTION_LENGTH'],
    'min_descriptions': CONFIG['MIN_DESCRIPTION_LENGTH'],
    'max_headings': CONFIG['MAX_HEADING_LENGTH'],
    'min_headings': 0,
}

def load_raw_data(filename):
    with open(filename, 'r') as fp:
        raw_data = json.load(fp)
    print(f'Loaded {len(raw_data):,} articles from {filename}')
    return raw_data

def tokenize_sentence(sentence):
    if CONFIG['TOKENIZER'] == 'nltk':
        return ' '.join(word_tokenize(sentence))
    elif CONFIG['TOKENIZER'] == 'custom':
        from custom_tokenizer import custom_tokenize
        return ' '.join(custom_tokenize(sentence))
    else:
        raise ValueError(f"Unsupported tokenizer: {CONFIG['TOKENIZER']}")

def article_is_complete(article: Dict) -> bool:
    """Check if an article has both heading and description."""
    return ('abstract' in article and 'article' in article 
            and article['abstract'] is not None and article['article'] is not None)

def tokenize_articles(raw_data: List[Dict]) -> Tuple[List[str], List[str]]:
    """Tokenize articles and create lists of headings and descriptions."""
    headings, descriptions = [], []
    
    for i, a in enumerate(raw_data):
        if article_is_complete(a):
            headings.append(tokenize_sentence(a['abstract']))
            descriptions.append(tokenize_sentence(a['article']))
        if i % 1000 == 0:  # Print progress every 1000 articles
            print(f'Tokenized {i:,} / {len(raw_data):,} articles')
    
    return headings, descriptions

def filter_text(text: str) -> str:
    """Filter out characters not in whitelist."""
    return ''.join(ch for ch in text if ch in WHITELIST)

def filter_length(headings: List[str], descriptions: List[str]) -> Tuple[List[str], List[str]]:
    """Filter articles based on length constraints."""
    if len(headings) != len(descriptions):
        raise ValueError('Number of headings does not match number of descriptions!')

    filtered_data = [
        (h, d) for h, d in zip(headings, descriptions)
        if (limit['min_descriptions'] <= len(d.split()) <= limit['max_descriptions'] and
            limit['min_headings'] <= len(h.split()) <= limit['max_headings'])
    ]

    filtered_headings, filtered_descriptions = zip(*filtered_data)
    
    print(f'Length of filtered headings: {len(filtered_headings):,}')
    print(f'Length of filtered descriptions: {len(filtered_descriptions):,}')

    return list(filtered_headings), list(filtered_descriptions)

def index_data(tokenized_sentences: List[List[str]], vocab_size: int) -> Tuple[List[str], Dict[str, int], nltk.FreqDist]:
    """Form vocabulary, idx2word and word2idx dictionaries."""
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    print(f'Vocab length: {len(vocab):,}')

    idx2word = ['_', UNK] + [x[0] for x in vocab]
    word2idx = {w: i for i, w in enumerate(idx2word)}

    return idx2word, word2idx, freq_dist

def pad_seq(seq: List[str], lookup: Dict[str, int], max_length: int) -> List[int]:
    """Pad sequence with zero values."""
    indices = [lookup.get(word, lookup[UNK]) for word in seq]
    return indices + [0] * (max_length - len(seq))

def zero_pad(tokenized_headings: List[List[str]], tokenized_descriptions: List[List[str]], word2idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Store indices in numpy arrays and create zero padding where required."""
    data_length = len(tokenized_descriptions)

    idx_descriptions = np.zeros((data_length, limit['max_descriptions']), dtype=np.int32)
    idx_headings = np.zeros((data_length, limit['max_headings']), dtype=np.int32)

    for i, (heading, description) in enumerate(zip(tokenized_headings, tokenized_descriptions)):
        idx_descriptions[i] = pad_seq(description, word2idx, limit['max_descriptions'])
        idx_headings[i] = pad_seq(heading, word2idx, limit['max_headings'])

    return idx_headings, idx_descriptions

def load_pretrained_embeddings(word2idx: Dict[str, int], embedding_dim: int = 300) -> np.ndarray:
    """Load pre-trained word embeddings."""
    model = KeyedVectors.load_word2vec_format('path_to_pretrained_embeddings', binary=True)
    embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    
    for word, i in word2idx.items():
        if word in model.key_to_index:
            embedding_matrix[i] = model[word]
    
    return embedding_matrix

def augment_data(descriptions: List[str]) -> List[str]:
    """Perform data augmentation on descriptions."""
    augmented_descriptions = []
    for description in descriptions:
        augmented_descriptions.append(description)
        # Add simple augmentation techniques
        augmented_descriptions.append(' '.join(np.random.permutation(description.split())))
        augmented_descriptions.append(' '.join(description.split()[::-1]))
    return augmented_descriptions

def process_data():
    """Process the data and prepare it for model training."""
    filename = CONFIG['RAW_DATA_FILE']
    raw_data = load_raw_data(filename)

    headings, descriptions = tokenize_articles(raw_data)

    headings = [filter_text(heading) for heading in headings]
    descriptions = [filter_text(sentence) for sentence in descriptions]
    headings, descriptions = filter_length(headings, descriptions)

    # Data augmentation
    if CONFIG['ENABLE_AUGMENTATION']:
        augmented_descriptions = augment_data(descriptions)
    else:
        augmented_descriptions = descriptions
    
    word_tokenized_headings = [word_list.split() for word_list in headings]
    word_tokenized_descriptions = [word_list.split() for word_list in augmented_descriptions]

    idx2word, word2idx, freq_dist = index_data(word_tokenized_headings + word_tokenized_descriptions, VOCAB_SIZE)

    idx_headings, idx_descriptions = zero_pad(word_tokenized_headings, word_tokenized_descriptions, word2idx)

    unk_percentage = calculate_unk_percentage(idx_headings, idx_descriptions, word2idx)
    print(f"UNK percentage: {unk_percentage:.2f}%")

    # Load pre-trained embeddings
    if CONFIG['USE_PRETRAINED_EMBEDDINGS']:
        embedding_matrix = load_pretrained_embeddings(word2idx)
    else:
        embedding_matrix = None

    article_data = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'limit': limit,
        'freq_dist': freq_dist,
        'embedding_matrix': embedding_matrix
    }

    save_data(article_data, idx_headings, idx_descriptions)

    return idx_headings, idx_descriptions

def save_data(article_data: Dict, idx_headings: np.ndarray, idx_descriptions: np.ndarray):
    """Save processed data to disk using HDF5 format."""
    with h5py.File(CONFIG['PROCESSED_DATA_FILE'], 'w') as hf:
        hf.create_dataset('idx_headings', data=idx_headings)
        hf.create_dataset('idx_descriptions', data=idx_descriptions)
        if article_data['embedding_matrix'] is not None:
            hf.create_dataset('embedding_matrix', data=article_data['embedding_matrix'])
        
        # Save metadata
        metadata = hf.create_group('metadata')
        metadata.attrs['vocab_size'] = len(article_data['word2idx'])
        metadata.attrs['max_heading_length'] = limit['max_headings']
        metadata.attrs['max_description_length'] = limit['max_descriptions']

    # Save other data using pickle
    with open(CONFIG['METADATA_FILE'], 'wb') as fp:
        pickle.dump({k: v for k, v in article_data.items() if k != 'embedding_matrix'}, fp)

def load_processed_data() -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Load processed data from disk."""
    with h5py.File(CONFIG['PROCESSED_DATA_FILE'], 'r') as hf:
        idx_headings = hf['idx_headings'][:]
        idx_descriptions = hf['idx_descriptions'][:]
        embedding_matrix = hf['embedding_matrix'][:] if 'embedding_matrix' in hf else None

    with open(CONFIG['METADATA_FILE'], 'rb') as fp:
        article_data = pickle.load(fp)
    
    article_data['embedding_matrix'] = embedding_matrix
    return article_data, idx_headings, idx_descriptions

def calculate_unk_percentage(idx_headings: np.ndarray, idx_descriptions: np.ndarray, word2idx: Dict[str, int]) -> float:
    """Calculate the percentage of unknown words in the dataset."""
    num_unk = np.sum(idx_headings == word2idx[UNK]) + np.sum(idx_descriptions == word2idx[UNK])
    num_words = np.sum(idx_headings > word2idx[UNK]) + np.sum(idx_descriptions > word2idx[UNK])
    return (num_unk / num_words) * 100

def main():
    process_data()

if __name__ == '__main__':
    main()