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

# Read the dataset
data = pd.read_csv('matchillioisTest.csv')
data['Student_Responses'] = data['Student_Responses'].apply(remove_stopwords)
texts = data['Student_Responses'].tolist()

# Preprocess and vectorize for LDA
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
dictionary = corpora.Dictionary.from_corpus(corpus, id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

# Fit LDA model
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=7, passes=10)
data['LDATopic'] = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(texts))]

# Fit BERTopic model
bertopic_model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size=5, nr_topics='auto')
bertopic_topics, _ = bertopic_model.fit_transform(texts)
data['BERTopic'] = bertopic_topics

# Sort the data first by LDA topic, then by BERTopic topic
data_sorted = data.sort_values(by=['LDATopic', 'BERTopic'])

# Save the sorted data to a CSV file
data_sorted.to_csv('Bert+LDA_sorted_student_responses_with_topics.csv', index=False)
