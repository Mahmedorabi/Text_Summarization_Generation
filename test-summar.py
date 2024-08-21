import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Title of the app
st.title("Text Summarization Generation")

# Text area for input
article_text = st.text_area("Paste or write your article text here", height=300)
# Get the number of sentences for the summary
number_of_summary = st.number_input(
    "Enter the number of sentences for summary:",
    min_value=1, max_value=15, value=5, step=1
)
if st.button('Summarize'):
    
    if article_text:
        # Split text into sentences
        sentences = sent_tokenize(article_text)
        
        # Text preprocessing
        cleaned_text = pd.Series(sentences).str.replace('[^a-zA-Z]', ' ', regex=True)
        cleaned_sentences = [s.lower() for s in cleaned_text]
        
        # Remove stopwords
        stop_words = stopwords.words('english')
        
        def remove_stopwords(sent):
            return ' '.join([i for i in sent.split() if i not in stop_words])
        
        cleaned_sentences = [remove_stopwords(r) for r in cleaned_sentences]
        
        # Load GloVe embeddings
        @st.cache_data(show_spinner=False)
        def load_glove_embeddings():
            file = open('D:\mohamed Orabi\glove.6B.100d.txt', encoding='utf-8')
            word_embedding = {}
            for line in file:
                value = line.split()
                word = value[0]
                coefs = np.asarray(value[1:], dtype='float32')
                word_embedding[word] = coefs
            file.close()
            return word_embedding
        
        word_embedding = load_glove_embeddings()
        
        # Sentence vectors
        sentence_vectors = []
        for i in cleaned_sentences:
            if len(i) != 0:
                vector=sum([word_embedding.get(w,np.zeros((100,)))for w in i.split()])/len(i.split())+0.001
            else:
                vector = np.zeros((100,))
            sentence_vectors.append(vector)
        
        # Similarity matrix
        similarity_matrix = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j]=cosine_similarity(sentence_vectors[i].reshape(1,100),
                                                              sentence_vectors[j].reshape(1, 100))[0, 0]
        
        # Graph-based ranking
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Ranking sentences
        rank_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        

        
        # Display the summary
        st.subheader("Summary:")
        for i in range(number_of_summary):
            st.write(rank_sentence[i][1])

