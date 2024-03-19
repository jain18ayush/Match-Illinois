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

def process_csv(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    data['Project Description'] = data['Project Description'].apply(remove_stopwords)
    texts = data['Project Description'].tolist()

    bertopic_model = BERTopic(verbose=True, embedding_model='all-MiniLM-L6-v2', min_topic_size=5)
    bertopic_topics, _ = bertopic_model.fit_transform(texts)
    data['BERTopic'] = bertopic_topics

    outlier_data = data[data['BERTopic'] == -1]
    outlier_texts = outlier_data['Project Description'].tolist()

    if outlier_texts:
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(outlier_texts)
        corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
        dictionary = corpora.Dictionary.from_corpus(corpus, id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
        outlier_data['LDATopic'] = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(outlier_texts))]

        data.loc[data['BERTopic'] == -1, 'LDATopic'] = outlier_data['LDATopic']

    sorted_data = data.sort_values(by=['BERTopic', 'LDATopic'])
    sorted_data.to_csv(output_csv, index=False)

# List of input and output files
files = [
    ('/Users/nicolehu/Documents/GitHub/Match-Illinois/Likert_Sorted_Groups.csv/App_Dev_group.csv', '/Users/nicolehu/Documents/GitHub/Match-Illinois/Bert_LDA_Sorted_Groups.csv/App_Dev_Sorted.csv'),
    ('/Users/nicolehu/Documents/GitHub/Match-Illinois/Likert_Sorted_Groups.csv/Data_Analytics_group.csv', '/Users/nicolehu/Documents/GitHub/Match-Illinois/Bert_LDA_Sorted_Groups.csv/Data_Analytics_Sorted.csv'),
    ('/Users/nicolehu/Documents/GitHub/Match-Illinois/Likert_Sorted_Groups.csv/ML_group.csv', '/Users/nicolehu/Documents/GitHub/Match-Illinois/Bert_LDA_Sorted_Groups.csv/ML_Sorted.csv'),
    ('/Users/nicolehu/Documents/GitHub/Match-Illinois/Likert_Sorted_Groups.csv/Mobile_Dev_group.csv', '/Users/nicolehu/Documents/GitHub/Match-Illinois/Bert_LDA_Sorted_Groups.csv/Mobile_Dev_Sorted.csv'),
    ('/Users/nicolehu/Documents/GitHub/Match-Illinois/Likert_Sorted_Groups.csv/Web_Dev_group.csv', '/Users/nicolehu/Documents/GitHub/Match-Illinois/Bert_LDA_Sorted_Groups.csv/Web_Dev_Sorted.csv')
]

for input_file, output_file in files:
    process_csv(input_file, output_file)
