## 📝 Comprehensive Text Summarization Project (Extractive vs. Abstractive)

This project is a detailed **Natural Language Processing (NLP)** study that compares and implements both **Extractive** and **Abstractive** text summarization methods. We utilize PyTorch and the Hugging Face T5 library to develop and evaluate five distinct summarization algorithms on English news articles.

---

### 🎯 Project Goals

* **Data Preparation:** Apply extensive cleaning, preprocessing, and tokenization to raw text data.
* **Extractive Methods:** Implement TF-IDF, TextRank, and Word2Vec-based algorithms to extract the most informative sentences from articles.
* **Abstractive Methods:** Train Seq2Seq (LSTM) and Seq2Seq + Attention models from scratch using PyTorch, and fine-tune the T5 large language model (LLM).
* **Comprehensive Evaluation:** Compare all algorithms using ROUGE, BLEU, METEOR scores, as well as speed and length metrics.

---

### 💾 Dataset Overview

The project uses a dataset consisting of English news articles and their human-written reference summaries. The data is categorized into five major domains: **business, entertainment, politics, sport, and tech**.

| Operation | Article | Summary |
| :--- | :--- | :--- |
| **Preprocessing** | Cleaning, Stopword removal, Stemming. | Cleaning, Stopword removal, Stemming. |
| **Tokenization** | Word-level tokens. | Word-level tokens, including `<sos>` and `<eos>` tags. |

---

### 🛠️ Extractive Methods 

Extractive summarization selects the most significant original sentences from the source text.

| Algorithm | Core Principle | Main Metric for Ranking |
| :--- | :--- | :--- |
| **TF-IDF** | Scores each sentence by the weighted average of rare and important words within the text. | Sentence Score Average. |
| **TextRank** | Constructs a graph where nodes are sentences and edges are cosine similarity. Sentence importance is determined by the PageRank algorithm. | PageRank Score (Node Importance). |
| **Word2Vec (Embedding)**| Uses the average of Word2Vec embeddings of words to create sentence vectors, which are then ranked by their mean similarity scores. | Sentence Vector Average Similarity. |

---

### 🤖 Abstractive Methods 

Abstractive summarization generates new sentences, producing more fluent and human-like summaries. 

#### 1. PyTorch-Based Models (Seq2Seq)

* **Vocabulary:** A single vocabulary (`word2idx`, `vocab_size`) is built from all articles and summaries, including special tokens like `<pad>`, `<unk>`, `<sos>`, `<eos>`.
* **Encoder:** Reads the input article and compresses the context into a hidden state vector (using a 4-layer LSTM).
* **Decoder:** Starts from the Encoder's hidden state and predicts summary words step-by-step.
* **Seq2Seq:** The basic Encoder-Decoder structure.
* **Seq2Seq + Attention:** An **Attention Mechanism** is added to the Decoder, allowing it to focus on relevant parts of the Encoder's output at every prediction step.

#### 2. Transformer-Based Model (LLM - T5)

* **Model:** `t5-small` (Hugging Face).
* **Method:** Fine-Tuning the pre-trained model for the summarization task.
* **Input Format:** `summarize: [article]`
* **Training:** 5 Epochs, Adam optimizer, executed on a GPU using the Hugging Face `Trainer` API.

---

### 📊 Algorithm Comparison and Evaluation 

All algorithms were comprehensively analyzed based on **Speed, Compression Ratio, and Quality** metrics.

#### Quality Metrics

| Metric Type | Metric(s) | Description |
| :--- | :--- | :--- |
| **Quality** | **ROUGE-1, ROUGE-2, ROUGE-L** | Measures the overlap (F-score) between the generated summary and the reference summary based on unigrams (1-gram), bigrams (2-gram), and the longest common subsequence. |
| **Semantic** | **BLEU** | Measures precision based on the overlap of $n$-grams between the prediction and reference. |
| **Semantic** | **METEOR** | A more advanced metric that considers synonymy, stemming, and weighted ordered word overlap. |
| **Performance** | **Processing Time** | Time taken to generate a summary for a single article. |
| **Output Features** | **Summary Length, Compression Ratio** | Word count of the output and the ratio of summary length to article length. |

<img width="769" height="344" alt="image" src="https://github.com/user-attachments/assets/c3421742-4ea6-41ab-9501-311b85283c49" />
<img width="769" height="344" alt="image" src="https://github.com/user-attachments/assets/200ac285-8596-45e2-823a-b7f86b19e3d5" />
<img width="769" height="344" alt="image" src="https://github.com/user-attachments/assets/0fcfc18a-0b34-4467-a8e7-096cc8b4c244" />
<img width="769" height="344" alt="image" src="https://github.com/user-attachments/assets/885d2596-441a-40ff-9cd5-cb60db83e434" />
<img width="769" height="344" alt="image" src="https://github.com/user-attachments/assets/b9e6481d-84ea-4652-9284-f667cc9a30d3" />
<img width="769" height="344" alt="image" src="https://github.com/user-attachments/assets/b3341e09-2a89-4939-bc49-2d24f85d629f" />
<img width="769" height="344" alt="image" src="https://github.com/user-attachments/assets/e83dc3d6-72d1-4fb7-84f8-d3392f6907aa" />






