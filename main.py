import streamlit as st
import pandas as pd
import numpy as np
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import pickle
import json
from datetime import datetime
import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
nltk.download('punkt')
nltk.download('punkt_tab')


# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TfidfSummarizer:
    def __init__(self, language="english", max_features=1000, ngram_range=(1, 1)):
        self.language = language
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = self.vectorizer_tfidf()

    def vectorizer_tfidf(self):
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            token_pattern=r'\b\w{2,}\b',
            stop_words=self.language)

    def summarize_text_tfidf(self, text, n=5):
        tokenize_sentences = sent_tokenize(text)
        tokenize_sentences = [s.strip() for s in tokenize_sentences if s.strip()]
        
        if len(tokenize_sentences) <= n:
            return ' '.join(tokenize_sentences)
            
        try:
            tfidf_matrix = self.vectorizer.fit_transform(tokenize_sentences)
            tfidf_score = np.mean(tfidf_matrix.toarray(), axis=1)
            
            sorted_scores = sorted(tfidf_score, reverse=True)[:n]
            top_sentences = []
            for i in range(len(tokenize_sentences)):
                if tfidf_score[i] in sorted_scores:
                    top_sentences.append(tokenize_sentences[i])
            
            summary = ' '.join(top_sentences)
            return summary
        except ValueError as e:
            if "empty vocabulary" in str(e):
                return ' '.join(tokenize_sentences[:n])
            else:
                raise

class TextRankSummarizer:
    def __init__(self, language="english", threshold=0.1):
        self.language = language
        self.threshold = threshold

    def text_rank_vectorizer(self):
        return TfidfVectorizer(stop_words=self.language, token_pattern=r'\b\w{2,}\b')

    def calculation_similarity(self, sentences):
        if not sentences:
            return np.array([[]])
        try:
            vectorizer = self.text_rank_vectorizer()
            tf_idf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tf_idf_matrix)
            
            np.fill_diagonal(similarity_matrix, 0)
            
            rows, cols = similarity_matrix.shape
            for i in range(rows):
                for j in range(cols):
                    if similarity_matrix[i, j] < self.threshold:
                        similarity_matrix[i, j] = 0
            return similarity_matrix
        except ValueError as e:
            if "empty vocabulary" in str(e):
                return np.array([[]])
            else:
                raise

    def text_rank_summary(self, text, num_sentences=5):
        token_sent = sent_tokenize(text)
        if len(token_sent) <= num_sentences:
            return text
            
        similarity_mat = self.calculation_similarity(token_sent)
        
        if similarity_mat.size == 0:
            return ' '.join(token_sent[:num_sentences])
            
        nx_graph = nx.from_numpy_array(similarity_mat)
        
        try:
            scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
        except:
            scores = {i: 1.0 for i in range(len(token_sent))}
        
        sorted_sentences = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_indices = [sorted_sentences[i][0] for i in range(min(num_sentences, len(sorted_sentences)))]
        top_indices.sort()
        
        summary_sentences = [token_sent[i] for i in top_indices]
        return ' '.join(summary_sentences)

class Word2VecSummarizer:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def create_corpus(self, texts):
        corpus = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            if len(tokens) > 3:
                corpus.append(tokens)
        return corpus

    def train_w2v_model(self, corpus):
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        return self.model

    def w2v_summary(self, text, num_sents=5):
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sents:
            return text

        corpus = self.create_corpus(sentences)
        if not corpus:
            return ' '.join(sentences[:num_sents])
            
        self.train_w2v_model(corpus)
        
        sentence_embeddings = []
        for sentence in sentences:
            token_words = word_tokenize(sentence.lower())
            word_vectors = []
            for word in token_words:
                try:
                    word_vectors.append(self.model.wv[word])
                except KeyError:
                    continue
            
            if word_vectors:
                sentence_embedding = np.mean(word_vectors, axis=0)
            else:
                sentence_embedding = np.zeros(self.model.wv.vector_size)
            sentence_embeddings.append(sentence_embedding)

        sentence_embeddings = np.array(sentence_embeddings)
        similarity_matrix = cosine_similarity(sentence_embeddings)
        scores = np.mean(similarity_matrix, axis=1)
        
        ranked_indices = np.argsort(scores)[::-1]
        selected_indices = sorted(ranked_indices[:num_sents])
        
        summary = " ".join([sentences[i] for i in selected_indices])
        return summary

class Seq2SeqTrainer:
    def __init__(self, vocab_size, word2idx, device, embedding_size=64, hidden_size=128, num_layers=5):
        self.vocab_size = vocab_size
        self.word2idx = word2idx
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.model = self._build_seq2seq()
        
        # Safe device transfer
        try:
            self.model.to(self.device)
        except NotImplementedError:
            # Handle meta tensor issue
            self.model = self.model.to_empty(device=self.device)
        except Exception:
            # Fallback: recreate model on target device
            self.model = self._build_seq2seq()
            for param in self.model.parameters():
                if param.is_meta:
                    param.data = torch.empty_like(param, device=self.device)
            self.model.to(self.device)

    def _build_encoder(self):
        class Encoder(nn.Module):
            def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx):
                super(Encoder, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
                self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layers)
                
            def forward(self, input_):
                embedded = self.embedding(input_)
                outputs, (hidden, cell) = self.lstm(embedded)
                return hidden, cell
                
        encoder = Encoder(self.vocab_size, self.embedding_size, self.hidden_size, self.num_layers, self.word2idx["<pad>"])
        return encoder

    def _build_decoder(self):
        class Decoder(nn.Module):
            def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx):
                super(Decoder, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
                self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_, hidden, cell):
                embedded = self.embedding(input_)
                output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
                predictions = self.fc(output)
                return predictions, hidden, cell
            
        decoder = Decoder(self.vocab_size, self.embedding_size, self.hidden_size, self.num_layers, self.word2idx["<pad>"])
        return decoder

    def _build_seq2seq(self):
        class Seq2Seq(nn.Module):
            def __init__(self, encoder, decoder, device):
                super(Seq2Seq, self).__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.device = device
                
            def forward(self, src, trg, teacher_forcing_ratio=0.5):
                batch_size = src.shape[0]
                trg_len = trg.shape[1]
                outputs = torch.zeros(batch_size, trg_len, self.decoder.fc.out_features).to(self.device)
                
                hidden, cell = self.encoder(src)
                decoder_input = trg[:, 0].unsqueeze(1)
                
                for t in range(1, trg_len):
                    output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                    outputs[:, t, :] = output.squeeze(1)
                    top1 = output.argmax(dim=2)
                    use_teacher_forcing = random.random() < teacher_forcing_ratio
                    decoder_input = trg[:, t].unsqueeze(1) if use_teacher_forcing else top1
                return outputs
                
        model = Seq2Seq(self.encoder, self.decoder, self.device)
        return model

class Seq2SeqTrainerAttention:
    def __init__(self, vocab_size, word2idx, device, embedding_size=64, hidden_size=128, num_layers=1):
        self.vocab_size = vocab_size
        self.word2idx = word2idx
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = self.AttentionEncoder(vocab_size, embedding_size, hidden_size, num_layers, padding_idx=word2idx["<pad>"])
        self.decoder = self.AttentionDecoder(vocab_size, embedding_size, hidden_size, num_layers, padding_idx=word2idx["<pad>"])
        self.model = self.AttentionSeq2Seq(self.encoder, self.decoder, device)
        
        # Safe device transfer
        try:
            self.model.to(self.device)
        except NotImplementedError:
            # Handle meta tensor issue
            self.model = self.model.to_empty(device=self.device)
        except Exception:
            # Fallback: recreate model on target device
            self.model = self.AttentionSeq2Seq(self.encoder, self.decoder, device)
            for param in self.model.parameters():
                if param.is_meta:
                    param.data = torch.empty_like(param, device=self.device)
            self.model.to(self.device)
        
    class AttentionEncoder(nn.Module):
        def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
            self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=False)
    
        def forward(self, input_):
            embedded = self.embedding(input_)
            outputs, (hidden, cell) = self.lstm(embedded)
            return outputs, hidden, cell

    class Attention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)
    
        def forward(self, hidden, encoder_outputs):
            hidden = hidden[-1].unsqueeze(1)
            seq_len = encoder_outputs.shape[1]
            hidden_expanded = hidden.repeat(1, seq_len, 1)
            energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
            attention = self.v(energy).squeeze(2)
            attention_weights = torch.softmax(attention, dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            return context, attention_weights

    class AttentionDecoder(nn.Module):
        def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
            self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size * 2, vocab_size)
            self.attention = Seq2SeqTrainerAttention.Attention(hidden_size)
    
        def forward(self, input_, hidden, cell, encoder_outputs):
            embedded = self.embedding(input_)
            context, attn_weights = self.attention(hidden, encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=2)
            output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
            prediction = self.fc(torch.cat((output, context), dim=2))
            return prediction, hidden, cell, attn_weights

    class AttentionSeq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
    
        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            batch_size = src.shape[0]
            trg_len = trg.shape[1]
            outputs = torch.zeros(batch_size, trg_len, self.decoder.fc.out_features).to(self.device)
            encoder_outputs, hidden, cell = self.encoder(src)
    
            decoder_input = trg[:, 0].unsqueeze(1)
            for t in range(1, trg_len):
                output, hidden, cell, attn_w = self.decoder(decoder_input, hidden, cell, encoder_outputs)
                outputs[:, t, :] = output.squeeze(1)
                top1 = output.argmax(dim=2)
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                decoder_input = trg[:, t].unsqueeze(1) if use_teacher_forcing else top1
            return outputs

def generate_seq2seq_summary(model, tokens, word2idx, idx2word, device, max_len=50):
    model.eval()
    with torch.no_grad():
        article_ids = [word2idx.get(w.lower(), word2idx["<unk>"]) for w in tokens]
        article_tensor = torch.tensor(article_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        hidden, cell = model.encoder(article_tensor)
        decoder_input = torch.tensor([[word2idx["<sos>"]]], device=device)
        summary_tokens = []
        
        for _ in range(max_len):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            top1 = output.argmax(2).item()
            if top1 == word2idx["<eos>"]:
                break
            summary_tokens.append(idx2word[top1])
            decoder_input = torch.tensor([[top1]], device=device)
        
        return " ".join(summary_tokens)

def generate_attention_summary(model, tokens, word2idx, idx2word, device, max_len=50):
    model.eval()
    with torch.no_grad():
        article_ids = [word2idx.get(w.lower(), word2idx["<unk>"]) for w in tokens]
        article_tensor = torch.tensor(article_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        encoder_outputs, hidden, cell = model.encoder(article_tensor)
        decoder_input = torch.tensor([[word2idx["<sos>"]]], device=device)
        summary_tokens = []
        
        for _ in range(max_len):
            output, hidden, cell, attn_weights = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            top1 = output.argmax(2).item()
            if top1 == word2idx["<eos>"]:
                break
            summary_tokens.append(idx2word[top1])
            decoder_input = torch.tensor([[top1]], device=device)
        
        return " ".join(summary_tokens)

def generate_t5_summary(model, tokenizer, text, device, max_len=64):
    input_ids = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids.to(device)
    summary_ids = model.generate(input_ids, max_length=max_len, min_length=10, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def compute_metrics(reference, prediction):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, prediction)
        
        bleu_score = sentence_bleu(
            [reference.split()], 
            prediction.split(), 
            smoothing_function=SmoothingFunction().method4
        )
        
        meteor = meteor_score([reference.split()], prediction.split())
        
        return {
            "ROUGE-1": round(rouge_scores["rouge1"].fmeasure, 4),
            "ROUGE-2": round(rouge_scores["rouge2"].fmeasure, 4),
            "ROUGE-L": round(rouge_scores["rougeL"].fmeasure, 4),
            "BLEU": round(bleu_score, 4),
            "METEOR": round(meteor, 4)
        }
    except Exception as e:
        return {
            "ROUGE-1": 0.0,
            "ROUGE-2": 0.0,
            "ROUGE-L": 0.0,
            "BLEU": 0.0,
            "METEOR": 0.0
        }

@st.cache_resource
def load_models():
    """Load all models with caching"""
    models = {}
    
    # Load vocabulary if exists
    try:
        with open("word2idx_fixed.pkl", "rb") as f:
            word2idx = pickle.load(f)
        idx2word = {i: w for w, i in word2idx.items()}
        
        # Load Seq2Seq model
        try:
            seq_trainer = Seq2SeqTrainer(vocab_size=len(word2idx), word2idx=word2idx, device=device, num_layers=5)
            seq_trainer.model.load_state_dict(torch.load("seq2seq_model.pth", map_location=device))
            seq_trainer.model.to(device)
            seq_trainer.model.eval()
            models['seq2seq'] = (seq_trainer, word2idx, idx2word)
            st.success("Seq2Seq model loaded successfully!")
        except Exception as e:
            st.warning(f"Seq2Seq model not found: {e}")
            models['seq2seq'] = None
        
        # Load Seq2Seq-Attention model
        try:
            attn_trainer = Seq2SeqTrainerAttention(
                vocab_size=len(word2idx), 
                word2idx=word2idx, 
                device=device, 
                num_layers=1
            )
            attn_trainer.model.load_state_dict(torch.load("attention_seq2seq_model.pth", map_location=device))
            attn_trainer.model.to(device)
            attn_trainer.model.eval()
            models['attention'] = (attn_trainer, word2idx, idx2word)
            st.success("Seq2Seq-Attention model loaded successfully!")
        except Exception as e:
            st.warning(f"Seq2Seq-Attention model not found: {e}")
            models['attention'] = None
            
    except Exception as e:
        st.warning(f"Vocabulary file not found: {e}")
        models['seq2seq'] = None
        models['attention'] = None
    
    # Load T5 model
    try:
        t5_model_path = r"t5_summarization\final_model"
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
        t5_model.to(device)
        t5_model.eval()
        models['t5'] = (t5_model, t5_tokenizer)
        st.success("T5 model loaded successfully!")
    except Exception as e:
        st.warning(f"T5 model not found: {e}")
        models['t5'] = None
    
    return models

def create_visualization(metrics_df, execution_times):
    """Create visualizations for metrics and execution times"""
    
    if metrics_df.empty:
        return None, None
    
    # Metrics comparison chart
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    metrics_to_plot = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'METEOR']
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    if available_metrics:
        metrics_df[available_metrics].plot(kind='bar', ax=ax1)
        ax1.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Algorithms', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    # Execution time chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    algorithms = list(execution_times.keys())
    times = list(execution_times.values())
    
    bars = ax2.bar(algorithms, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('Algorithm Execution Times', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Algorithms', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig1, fig2

def create_word_cloud(text, title="Word Cloud"):
    """Create a word cloud from text"""
    try:
        # Clean text
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        
        if not words:
            return None
            
        clean_text = ' '.join(words)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(clean_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Word cloud oluşturulurken hata: {e}")
        return None

def export_results(summaries, metrics, execution_times):
    """Export results to JSON and CSV"""
    
    # Prepare data for export
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "summaries": summaries,
        "execution_times": execution_times
    }
    
    if metrics:
        export_data["metrics"] = metrics
    
    # JSON export
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    # CSV export for metrics
    csv_data = None
    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        csv_data = metrics_df.to_csv(index=True)
    
    return json_str, csv_data

def main():
    st.set_page_config(page_title="Comprehensive Text Summarization", layout="wide")
    
    st.title("📄 Kapsamlı Metin Özetleme Sistemi")
    st.markdown("6 farklı algoritma ile metin özetleme ve karşılaştırma")
    st.markdown("---")
    
    # Load models
    with st.spinner("Modeller yükleniyor..."):
        models = load_models()
    
    # Sidebar for algorithm selection
    st.sidebar.header("⚙️ Ayarlar")
    
    # Algorithm selection
    algorithms = {
        "TF-IDF": True,
        "TextRank": True,
        "Word2Vec": True,
        "Seq2Seq": models['seq2seq'] is not None,
        "Seq2Seq-Attention": models['attention'] is not None,
        "T5": models['t5'] is not None
    }
    
    st.sidebar.subheader("🔧 Algoritma Seçimi")
    selected_algorithms = {}
    for algo, available in algorithms.items():
        if available:
            selected_algorithms[algo] = st.sidebar.checkbox(f"{algo}", value=True)
        else:
            st.sidebar.checkbox(f"{algo} (Model not found)", value=False, disabled=True)
            selected_algorithms[algo] = False
    
    # Parameters
    num_sentences = st.sidebar.slider("📊 Özet Cümle Sayısı (Klasik Algoritmalar):", 1, 10, 3)
    max_length_neural = st.sidebar.slider("📏 Maksimum Uzunluk (Neural Modeller):", 20, 100, 50)
    
    # Advanced settings
    with st.sidebar.expander("🔧 Gelişmiş Ayarlar"):
        show_word_cloud = st.checkbox("Word Cloud Göster", value=True)
        show_visualizations = st.checkbox("Görselleştirmeleri Göster", value=True)
        export_results_option = st.checkbox("Sonuçları Dışa Aktar", value=False)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Metin Girişi")
        
        # Text input options
        input_method = st.radio(
            "Metin giriş yöntemi seçin:",
            ("Metin Kutusu", "Dosya Yükleme"),
            horizontal=True
        )
        
        input_text = ""
        
        if input_method == "Metin Kutusu":
            input_text = st.text_area(
                "Özetlemek istediğiniz metni buraya girin:",
                height=300,
                placeholder="Metninizi buraya yapıştırın..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Metin dosyası yükleyin:",
                type=['txt', 'md'],
                help="Sadece .txt ve .md dosyaları desteklenir"
            )
            
            if uploaded_file is not None:
                try:
                    input_text = str(uploaded_file.read(), "utf-8")
                    st.success(f"Dosya başarıyla yüklendi! ({len(input_text)} karakter)")
                except Exception as e:
                    st.error(f"Dosya okuma hatası: {e}")
        
        reference_summary = st.text_area(
            "📋 Referans Özet (Karşılaştırma için - opsiyonel):",
            height=100,
            placeholder="Karşılaştırma yapmak için gerçek özeti buraya girin..."
        )
    
    with col2:
        st.subheader("📊 İstatistikler")
        if input_text:
            word_count = len(input_text.split())
            sentence_count = len(sent_tokenize(input_text))
            char_count = len(input_text)
            
            st.metric("Kelime Sayısı", word_count)
            st.metric("Cümle Sayısı", sentence_count)
            st.metric("Karakter Sayısı", char_count)
            
            # Additional statistics
            avg_words_per_sentence = round(word_count / sentence_count, 2) if sentence_count > 0 else 0
            
            with st.expander("📈 Detaylı İstatistikler"):
                st.write(f"**Ortalama kelime/cümle:** {avg_words_per_sentence}")
                st.write(f"**Ortalama karakter/kelime:** {round(char_count / word_count, 2) if word_count > 0 else 0}")
                
                # Word frequency
                words = word_tokenize(input_text.lower())
                words = [w for w in words if w.isalpha()]
                word_freq = Counter(words).most_common(5)
                
                st.write("**En sık kullanılan kelimeler:**")
                for word, freq in word_freq:
                    st.write(f"- {word}: {freq}")
        
        st.subheader("🤖 Model Durumu")
        model_status = {
            "Seq2Seq": models['seq2seq'] is not None,
            "Attention": models['attention'] is not None,
            "T5": models['t5'] is not None
        }
        
        for model_name, status in model_status.items():
            if status:
                st.success(f"✅ {model_name}")
            else:
                st.error(f"❌ {model_name}")

    # Sample texts for testing
    st.subheader("📚 Örnek Metinler")
    with st.expander("Örnek metinlerden birini seçin"):
        sample_texts = {
            "Teknoloji Makalesi": """
            Artificial intelligence has revolutionized numerous industries over the past decade. Machine learning algorithms
            have become increasingly sophisticated, enabling computers to perform tasks that were once thought to be exclusively
            human. Natural language processing has advanced significantly, allowing machines to understand and generate human
            language with remarkable accuracy. Computer vision technologies have also made tremendous progress, enabling
            applications in autonomous vehicles, medical diagnosis, and surveillance systems. The integration of AI into
            everyday life continues to accelerate, with smart assistants, recommendation systems, and automated decision-making
            becoming commonplace. However, these advancements also raise important questions about privacy, employment, and
            the ethical implications of artificial intelligence. As we move forward, it is crucial to develop AI systems
            that are transparent, fair, and beneficial to society as a whole.
            """,
            
            "Bilim Makalesi": """
            Climate change represents one of the most pressing challenges of our time. Rising global temperatures have led
            to melting ice caps, rising sea levels, and increasingly extreme weather patterns. The primary driver of this
            change is the increased concentration of greenhouse gases in the atmosphere, particularly carbon dioxide from
            fossil fuel combustion. Scientific consensus indicates that immediate action is required to mitigate the worst
            effects of climate change. Renewable energy sources such as solar, wind, and hydroelectric power offer promising
            alternatives to fossil fuels. Additionally, energy efficiency improvements, carbon capture technologies, and
            changes in land use practices can contribute to reducing greenhouse gas emissions. International cooperation
            and policy coordination are essential for addressing this global challenge effectively.
            """,
            
            "Eğitim Metni": """
            Education systems around the world are undergoing significant transformation in the digital age. Traditional
            classroom-based learning is being supplemented and sometimes replaced by online and hybrid learning models.
            Educational technology has introduced new tools and platforms that enable personalized learning experiences
            tailored to individual student needs. Virtual reality and augmented reality are beginning to create immersive
            learning environments that can simulate real-world scenarios. Artificial intelligence is being used to provide
            intelligent tutoring systems and automated assessment tools. However, the digital divide remains a significant
            challenge, as not all students have equal access to technology and high-speed internet. Educators must also
            adapt their teaching methods to effectively integrate these new technologies while maintaining the human
            connection that is essential for effective learning.
            """
        }
        
        selected_sample = st.selectbox("Örnek metin seçin:", [""] + list(sample_texts.keys()))
        if selected_sample and st.button("Örnek Metni Yükle"):
            input_text = sample_texts[selected_sample]
            st.rerun()

    if st.button("🚀 Özetleme Başlat", type="primary") and input_text.strip():
        if not any(selected_algorithms.values()):
            st.error("Lütfen en az bir algoritma seçin!")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        summaries = {}
        metrics = {}
        execution_times = {}
        
        selected_algos = [algo for algo, selected in selected_algorithms.items() if selected]
        total_algos = len(selected_algos)
        
        for i, algorithm in enumerate(selected_algos):
            status_text.text(f"🔄 {algorithm} algoritması çalışıyor...")
            start_time = time.time()
            
            try:
                if algorithm == "TF-IDF":
                    summarizer = TfidfSummarizer()
                    summaries[algorithm] = summarizer.summarize_text_tfidf(input_text, n=num_sentences)
                    
                elif algorithm == "TextRank":
                    summarizer = TextRankSummarizer()
                    summaries[algorithm] = summarizer.text_rank_summary(input_text, num_sentences=num_sentences)
                    
                elif algorithm == "Word2Vec":
                    summarizer = Word2VecSummarizer()
                    summaries[algorithm] = summarizer.w2v_summary(input_text, num_sents=num_sentences)
                
                elif algorithm == "Seq2Seq" and models['seq2seq']:
                    seq_trainer, word2idx, idx2word = models['seq2seq']
                    tokens = word_tokenize(input_text.lower())
                    summaries[algorithm] = generate_seq2seq_summary(
                        seq_trainer.model, tokens, word2idx, idx2word, device, max_length_neural
                    )
                
                elif algorithm == "Seq2Seq-Attention" and models['attention']:
                    attn_trainer, word2idx, idx2word = models['attention']
                    tokens = word_tokenize(input_text.lower())
                    summaries[algorithm] = generate_attention_summary(
                        attn_trainer.model, tokens, word2idx, idx2word, device, max_length_neural
                    )
                
                elif algorithm == "T5" and models['t5']:
                    t5_model, t5_tokenizer = models['t5']
                    summaries[algorithm] = generate_t5_summary(
                        t5_model, t5_tokenizer, input_text, device, max_length_neural
                    )
                
                execution_times[algorithm] = round(time.time() - start_time, 3)
                
                # Compute metrics if reference summary is provided
                if reference_summary.strip():
                    metrics[algorithm] = compute_metrics(reference_summary, summaries[algorithm])
                    metrics[algorithm]["Execution_Time"] = execution_times[algorithm]
                
            except Exception as e:
                st.error(f"❌ {algorithm} algoritmasında hata: {str(e)}")
                summaries[algorithm] = "Hata oluştu"
                execution_times[algorithm] = 0
            
            progress_bar.progress((i + 1) / total_algos)
        
        status_text.text("✅ Tüm özetlemeler tamamlandı!")
        
        # Display results
        st.markdown("---")
        st.header("📋 Sonuçlar")
        
        # Display summaries
        st.subheader("📝 Üretilen Özetler")
        
        # Create tabs for better organization
        if len(summaries) > 3:
            tab1, tab2 = st.tabs(["Klasik Algoritmalar", "Neural Modeller"])
            
            with tab1:
                for algo in ["TF-IDF", "TextRank", "Word2Vec"]:
                    if algo in summaries:
                        with st.expander(f"🔍 {algo} Özeti", expanded=True):
                            st.write(summaries[algo])
                            st.caption(f"⏱️ Çalışma süresi: {execution_times.get(algo, 0)} saniye")
                            
                            # Summary statistics
                            summary_words = len(summaries[algo].split())
                            summary_chars = len(summaries[algo])
                            original_words = len(input_text.split())
                            compression_ratio = round((1 - summary_words/original_words) * 100, 1) if original_words > 0 else 0
                            
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("Kelime", summary_words)
                            with col_stat2:
                                st.metric("Karakter", summary_chars)
                            with col_stat3:
                                st.metric("Sıkıştırma %", compression_ratio)
            
            with tab2:
                for algo in ["Seq2Seq", "Seq2Seq-Attention", "T5"]:
                    if algo in summaries:
                        with st.expander(f"🔍 {algo} Özeti", expanded=True):
                            st.write(summaries[algo])
                            st.caption(f"⏱️ Çalışma süresi: {execution_times.get(algo, 0)} saniye")
                            
                            # Summary statistics
                            summary_words = len(summaries[algo].split())
                            summary_chars = len(summaries[algo])
                            original_words = len(input_text.split())
                            compression_ratio = round((1 - summary_words/original_words) * 100, 1) if original_words > 0 else 0
                            
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("Kelime", summary_words)
                            with col_stat2:
                                st.metric("Karakter", summary_chars)
                            with col_stat3:
                                st.metric("Sıkıştırma %", compression_ratio)
        else:
            for algorithm, summary in summaries.items():
                with st.expander(f"🔍 {algorithm} Özeti", expanded=True):
                    st.write(summary)
                    st.caption(f"⏱️ Çalışma süresi: {execution_times.get(algorithm, 0)} saniye")
                    
                    # Summary statistics
                    summary_words = len(summary.split())
                    summary_chars = len(summary)
                    original_words = len(input_text.split())
                    compression_ratio = round((1 - summary_words/original_words) * 100, 1) if original_words > 0 else 0
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Kelime", summary_words)
                    with col_stat2:
                        st.metric("Karakter", summary_chars)
                    with col_stat3:
                        st.metric("Sıkıştırma %", compression_ratio)
        
        # Display metrics if reference summary provided
        if reference_summary.strip() and metrics:
            st.subheader("📊 Performans Metrikleri")
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(metrics).T
            
            # Display metrics table
            st.dataframe(metrics_df.round(4), use_container_width=True)
            
            # Best performing algorithm
            if not metrics_df.empty:
                st.subheader("🏆 En İyi Performans")
                
                best_rouge1 = metrics_df['ROUGE-1'].idxmax()
                best_rouge2 = metrics_df['ROUGE-2'].idxmax()
                best_rougeL = metrics_df['ROUGE-L'].idxmax()
                best_bleu = metrics_df['BLEU'].idxmax()
                best_meteor = metrics_df['METEOR'].idxmax()
                
                col_best1, col_best2, col_best3 = st.columns(3)
                with col_best1:
                    st.info(f"**ROUGE-1:** {best_rouge1}")
                    st.info(f"**ROUGE-2:** {best_rouge2}")
                with col_best2:
                    st.info(f"**ROUGE-L:** {best_rougeL}")
                    st.info(f"**BLEU:** {best_bleu}")
                with col_best3:
                    st.info(f"**METEOR:** {best_meteor}")
        
        # Visualizations
        if show_visualizations and metrics:
            st.subheader("📈 Görselleştirmeler")
            
            metrics_df = pd.DataFrame(metrics).T
            fig1, fig2 = create_visualization(metrics_df, execution_times)
            
            if fig1 is not None:
                st.pyplot(fig1)
            if fig2 is not None:
                st.pyplot(fig2)
        
        # Word cloud
        if show_word_cloud and input_text:
            st.subheader("☁️ Kelime Bulutu")
            
            col_wc1, col_wc2 = st.columns(2)
            
            with col_wc1:
                st.write("**Orijinal Metin**")
                wc_original = create_word_cloud(input_text, "Original Text Word Cloud")
                if wc_original:
                    st.pyplot(wc_original)
            
            with col_wc2:
                if summaries:
                    st.write("**Özetler (Birleşik)**")
                    combined_summaries = " ".join(summaries.values())
                    wc_summary = create_word_cloud(combined_summaries, "Summaries Word Cloud")
                    if wc_summary:
                        st.pyplot(wc_summary)
        
        # Export functionality
        if export_results_option and summaries:
            st.subheader("💾 Sonuçları Dışa Aktar")
            
            json_data, csv_data = export_results(summaries, metrics, execution_times)
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.download_button(
                    label="📄 JSON olarak indir",
                    data=json_data,
                    file_name=f"summarization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col_export2:
                if csv_data:
                    st.download_button(
                        label="📊 Metrikleri CSV olarak indir",
                        data=csv_data,
                        file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Comparison analysis
        if len(summaries) > 1:
            st.subheader("🔍 Karşılaştırmalı Analiz")
            
            # Length comparison
            st.write("**Özet Uzunlukları:**")
            length_data = []
            for algo, summary in summaries.items():
                length_data.append({
                    "Algoritma": algo,
                    "Kelime Sayısı": len(summary.split()),
                    "Karakter Sayısı": len(summary),
                    "Çalışma Süresi (s)": execution_times.get(algo, 0)
                })
            
            length_df = pd.DataFrame(length_data)
            st.dataframe(length_df, use_container_width=True)
            
            # Similarity analysis between summaries
            st.write("**Özetler Arası Benzerlik:**")
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                summary_texts = list(summaries.values())
                tfidf_matrix = vectorizer.fit_transform(summary_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=summaries.keys(),
                    columns=summaries.keys()
                )
                
                # Create heatmap
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    similarity_df,
                    annot=True,
                    cmap='YlOrRd',
                    center=0.5,
                    ax=ax_heatmap,
                    cbar_kws={'label': 'Cosine Similarity'}
                )
                ax_heatmap.set_title('Summary Similarity Heatmap', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_heatmap)
                
            except Exception as e:
                st.error(f"Benzerlik analizi hatası: {e}")

    # Information and help section
    st.markdown("---")
    with st.expander("ℹ️ Algoritmalar Hakkında Bilgi"):
        st.markdown("""
        ### 🔧 Kullanılan Algoritmalar:
        
        **1. TF-IDF (Term Frequency-Inverse Document Frequency):**
        - Kelimelerin önemini belirlemek için istatistiksel yöntem kullanır
        - Hızlı ve etkili, kısa metinler için idealdir
        
        **2. TextRank:**
        - Google'ın PageRank algoritmasından esinlenmiştir
        - Cümleler arası ilişkileri analiz eder
        - Uzun metinler için daha etkilidir
        
        **3. Word2Vec:**
        - Kelimelerin vektör temsillerini kullanır
        - Anlamsal benzerlik yakalayabilir
        - Orta uzunluktaki metinler için uygun
        
        **4. Seq2Seq (Sequence-to-Sequence):**
        - LSTM tabanlı encoder-decoder mimarisi
        - End-to-end öğrenme
        - Daha yaratıcı özetler üretebilir
        
        **5. Seq2Seq with Attention:**
        - Attention mekanizması ile geliştirilmiş Seq2Seq
        - Uzun metinlerde daha iyi performans
        - Hangi kısımlara odaklandığını gösterir
        
        **6. T5 (Text-to-Text Transfer Transformer):**
        - Transformer mimarisi kullanır
        - En son teknoloji
        - Yüksek kaliteli özetler üretir
        """)
    
    with st.expander("📊 Metriklerin Açıklaması"):
        st.markdown("""
        ### 📈 Performans Metrikleri:
        
        **ROUGE Skorları:**
        - **ROUGE-1:** Tekli kelime örtüşmesi (1-gram)
        - **ROUGE-2:** İkili kelime örtüşmesi (2-gram)
        - **ROUGE-L:** En uzun ortak alt dizi
        
        **BLEU Skoru:**
        - N-gram hassasiyetinin harmonik ortalaması
        - Makine çevirisi değerlendirmesinden uyarlanmış
        
        **METEOR Skoru:**
        - BLEU'nun geliştirilmiş versiyonu
        - Kök kelime ve eş anlamlı kelimeleri dikkate alır
        
        **Not:** Tüm skorlar 0-1 arasında, yüksek değerler daha iyi performansı gösterir.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>📄 Kapsamlı Metin Özetleme Sistemi | Geliştirildi: Python, Streamlit, PyTorch, Transformers</p>
            <p><small>Sistem bilgileri: GPU kullanımı {}, PyTorch {}</small></p>
        </div>
        """.format(
            "Aktif" if torch.cuda.is_available() else "Pasif",
            torch.__version__
        ),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()