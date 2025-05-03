import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer
import nltk
from bert_score import score as bert_score

# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# Title of the app
st.title("Génération de Résumé de Texte")

# Text area for input
article_text = st.text_area("Collez ou écrivez votre texte ici", height=300)

# Get the number of sentences for the summary
number_of_summary = st.number_input(
    "Nombre de phrases pour le résumé:",
    min_value=1, max_value=15, value=5, step=1
)

if st.button('Résumer'):
    if article_text:
        with st.spinner('Génération du résumé en cours...'):
            # Split text into sentences
            sentences = sent_tokenize(article_text, language='french')
            
            if len(sentences) < number_of_summary:
                st.warning(f"Le texte contient seulement {len(sentences)} phrases. Réduisez le nombre de phrases pour le résumé.")
                number_of_summary = len(sentences)
            
            # Text preprocessing - keep punctuation for better embedding context
            cleaned_sentences = [s.lower() for s in sentences]
            
            # Remove stopwords
            stop_words = set(stopwords.words('french'))
            
            def remove_stopwords(sent):
                return ' '.join([i for i in sent.split() if i not in stop_words])
            
            cleaned_sentences_no_stopwords = [remove_stopwords(r) for r in cleaned_sentences]
            
            # Load a multilingual embedding model with good French support
            @st.cache_resource(show_spinner=False)
            def load_french_embeddings():
                # This model has excellent French language support
                return SentenceTransformer("distiluse-base-multilingual-cased-v2")
            
            model = load_french_embeddings()
            
            # Generate embeddings for each sentence
            # Use the original sentences for better embedding quality
            sentence_vectors = model.encode(cleaned_sentences)
            
            # Similarity matrix
            similarity_matrix = np.zeros([len(sentences), len(sentences)])
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        similarity_matrix[i][j] = cosine_similarity(
                            sentence_vectors[i].reshape(1, -1),
                            sentence_vectors[j].reshape(1, -1)
                        )[0, 0]
            
            # Graph-based ranking using PageRank
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Rank sentences
            ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
            
            # Sort by original position to maintain flow
            summary_sentences = sorted(ranked_sentences[:number_of_summary], key=lambda x: x[2])
            
            # Display the summary
            st.subheader("Résumé:")
            summary_text = ""
            for _, sentence, _ in summary_sentences:
                summary_text += sentence + " "
                
            st.write(summary_text)
            st.download_button(
                label="Télécharger le résumé",
                data=summary_text,
                file_name="resume.txt",
                mime="text/plain"
            )
            
            # Show statistics
            st.subheader("Statistiques:")
            original_word_count = len(article_text.split())
            summary_word_count = len(summary_text.split())
            reduction = (1 - (summary_word_count / original_word_count)) * 100
            P, R, F1 = bert_score([summary_text], [article_text], lang="fr", rescale_with_baseline=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mots dans le texte original", original_word_count)
            col2.metric("Mots dans le résumé", summary_word_count)
            col3.metric("Réduction", f"{reduction:.1f}%")
            col4.metric("BERTScore F1", f"{F1[0].item() * 100:.2f}%")
