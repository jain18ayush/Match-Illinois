import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import gensim
from bertopic import BERTopic
import warnings

warnings.filterwarnings("ignore")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    clean_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(clean_words)

# Read and preprocess the dataset
data = pd.read_csv('matchillioisTest.csv')
data['Student_Responses'] = data['Student_Responses'].apply(remove_stopwords)
texts = data['Student_Responses'].tolist()

# Fit BERTopic model
bertopic_model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size=5)
bertopic_topics, _ = bertopic_model.fit_transform(texts)
data['BERTopic'] = bertopic_topics

# Filter out outliers marked with -1 by BERTopic
outlier_data = data[data['BERTopic'] == -1]
outlier_texts = outlier_data['Student_Responses'].tolist()

# Preprocess and vectorize texts for LDA
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(outlier_texts)
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
dictionary = corpora.Dictionary.from_corpus(corpus, id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

# Fit LDA model on the outliers
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
outlier_data['LDATopic'] = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(outlier_texts))]

# Combine the LDA topic assignments for outliers with the original dataset
data.loc[data['BERTopic'] == -1, 'LDATopic'] = outlier_data['LDATopic']

# Sort the data first by BERTopic (excluding -1), then by LDATopic for outliers
non_outlier_data = data[data['BERTopic'] != -1].sort_values(by='BERTopic')
outlier_data_sorted = outlier_data.sort_values(by='LDATopic')
sorted_data = pd.concat([non_outlier_data, outlier_data_sorted])

# Save the sorted data to a CSV file
sorted_data.to_csv('sorted_bertopic_and_lda_combined_responses.csv', index=False)
