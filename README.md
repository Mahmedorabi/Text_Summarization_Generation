# Text Summarization Using Glove Embeddings
## Overview
This notebook demonstrates the implementation of a text summarization algorithm using pre-trained Glove word embeddings and cosine similarity. The goal is to extract the most relevant sentences from a given set of articles to create a summary.

## Dependencies
The following libraries are required to run the notebook:

 - pandas
 - numpy
 - nltk
 - scikit-learn
 - networkx
Make sure to install these packages using pip if they are not already installed.

## Dataset
The dataset used in this project consists of articles related to tennis. The dataset is loaded from a CSV file.

## Methodology
The text summarization process involves several key steps:

**1. Reading the Dataset:** <br>
The dataset is loaded into a pandas DataFrame.

**2. Sentence Tokenization:**<br>
The articles are split into individual sentences using the sent_tokenize method from NLTK.

**3. Text Preprocessing:**
 - ****Removing Punctuation and Special Characters****: Non-alphabetic characters are removed from the sentences.
 - ****Lowercasing****: All text is converted to lowercase.
 - ****Stopwords Removal****: Common English stopwords are removed using NLTK's stopwords list.
 - ****Word Embedding****:<br>
Pre-trained Glove word embeddings (glove.6B.100d.txt) are used to convert words into vectors. Each sentence is represented by the average of its word vectors.

 - ****Similarity Matrix Calculation:****<br>
A cosine similarity matrix is generated to measure the similarity between sentences.

 - ****Graph Representation and Ranking:****<br>
A graph is constructed using NetworkX, where each node represents a sentence, and edges represent the similarity scores. The PageRank algorithm is applied to rank the sentences.

 - ****Summary Generation:****<br>
The top-ranked sentences are selected to form the summary. The number of sentences in the summary is determined by user input.

## Usage
**1. Load the Dataset:**<br>
Replace the file path in the code with the path to your dataset.

**2. Preprocess the Text:**
The notebook will automatically clean the text and prepare it for embedding.

**3. Generate Summary:**
Run the cells to generate a summary. The number of sentences in the summary can be specified by the user.

## Notes
- The Glove embeddings file (glove.6B.100d.txt) must be available in the working directory or specified path.
- The input data must be in CSV format with an 'article_text' column.
- Extracting Word Vector(link for download a glove)
  https://www.kaggle.com/datasets/wiseyy/glove6b100dtxt

