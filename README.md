# Article Data Processing Script

This script is designed to process raw article data, including tokenization, filtering, and conversion to numerical representations suitable for machine learning tasks. The script leverages the Natural Language Toolkit (nltk) for tokenization and frequency distribution analysis, and NumPy for numerical operations.

## Dependencies

Before running this script, ensure the following dependencies are installed:

- `nltk`: For natural language processing tasks.
- `json`: For handling JSON data.
- `config`: A custom configuration module for the project.
- `numpy`: For numerical operations.
- `cPickle` (or `pickle` in Python 3): For serializing objects.

## Configuration

The script assumes the existence of a `config` module with a `path_data` variable that points to the directory containing the raw data and where the processed data will be stored.

## Usage

1. Place your raw article data in a JSON file named `raw_data.json` in the `path_data` directory.
2. Run the script using `python analitika.py
3. The processed data will be saved in the `path_data` directory as a pickle file named `article_data.pkl`.

## Description

### Functions

- `load_raw_data(filename)`: Loads the raw JSON data and returns a list of articles.
- `tokenize_sentence(sentence)`: Tokenizes a sentence into words and returns a string of space-separated words.
- `article_is_complete(article)`: Checks if an article has both a heading and a description.
- `tokenize_articles(raw_data)`: Tokenizes the articles and returns lists of headings and descriptions.
- `filter(line, whitelist)`: Filters out characters not in the whitelist from a line of text.
- `filter_length(headings, descriptions)`: Filters headings and descriptions based on length limits.
- `index_data(tokenized_sentences, vocab_size)`: Creates a vocabulary, converts words to indices, and returns necessary data structures.
- `pad_seq(seq, lookup, max_length)`: Pads a sequence with zeros to the maximum length.
- `zero_pad(tokenized_headings, tokenized_descriptions, word2idx)`: Converts tokenized data to numpy arrays with zero padding.
- `process_data()`: Orchestrates the data processing pipeline.
- `pickle_data(article_data)`: Saves processed data to disk as a pickle file.
- `unpickle_articles()`: Loads processed data from disk.
- `calculate_unk_percentage(idx_headings, idx_descriptions, word2idx)`: Calculates the percentage of unknown words in the data.
- `main()`: The entry point of the script that calls `process_data()`.

### Global Variables

- `WHITELIST`: A string of characters allowed in the processed text.
- `VOCAB_SIZE`: The maximum size of the vocabulary.
- `UNK`: The token used to represent unknown words.
- `limit`: A dictionary containing length limits for headings and descriptions.

## Notes

- The script assumes that each article in the raw data has `abstract` and `article` fields for headings and descriptions, respectively.
- The processed data includes a vocabulary, word-to-index and index-to-word mappings, and frequency distributions of words.
- The script performs zero padding on tokenized data to ensure uniform lengths for batch processing in machine learning models.
- The percentage of unknown words in the final dataset is calculated and printed for analysis.

## License

This script is provided as-is, without warranty. It is free for personal and commercial use under the MIT License.
