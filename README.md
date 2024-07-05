# 🚀 Article Data Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

This is a really old script from IBM Watson times I'm trying to keep afresh :)

## 🌟 So, what it does?

- 🎯 **Efficient Processing**: Tokenize and filter articles (now ***a little bit*** faster)
- 🧠 **Pre-trained Embeddings**:
- 🔮 **Data Augment**: Expand your dataset
- 💾 **Storage**: I/O HDF5
- 🛠 **Customizable**: Hopefully! ;)

## 🚀 Quick Start

1. Clone the repo:
   ```
   git clone https://github.com/0101011/analitika.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the script:
   ```
   python analitika.py
   ```

## 📚 Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

## 🎮 Usage

1. Place your `raw_data.json` in the `data/` directory
2. (Optional) Add pre-trained embeddings to `data/`
3. Run the script:
   ```
   python analitika.py
   ```
4. Find processed data in `data/` as HDF5 and pickle files

## ⚙ Configuration

Customize the script by modifying these variables:

- `WHITELIST`: Allowed characters
- `VOCAB_SIZE`: Maximum vocabulary size
- `limit`: Length constraints for articles

## 🤝 Contributing

Here are some ways you can contribute:

- 💡 My goal was to develop a package or CLI tool out of it. Maybe we'll come up with something.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgements

- [NLTK](https://www.nltk.org/) for natural language processing
- [Gensim](https://radimrehurek.com/gensim/) for word embeddings
- [HDF5 for Python](https://www.h5py.org/) for efficient data storage

---

<p align="center">
  Made with ❤️ by [Your Name]
</p>
