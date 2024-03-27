import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import gensim
from bertopic import BERTopic
import warnings
from sklearn.cluster import KMeans


# PROBABILITY "LIKERT" --------------------------------------------------------
# Load the CSV data
data = pd.read_csv("data.csv")
list_csv_files = []

# Extract relevant columns (likert rankings)
likert_columns = ["Web Dev", "ML", "Mobile Dev", "App Dev", "Data Analytics"]

# Define probability weights based on likert rankings
weights = {
    1: 0.5,  # Higher weight for likert 1
    2: 0.4,  # Higher weight for likert 2
    3: 0.05,
    4: 0.03,
    5: 0.02
}

# Convert likert rankings to probabilities
probabilities = data[likert_columns].apply(lambda x: [weights[val] for val in x], axis=1)

# Assign users to groups probabilistically based on likert rankings
group_assignments = {}
for idx, topic_probabilities in probabilities.items():
    user_id = data.loc[idx, 'User']
    assigned_topic_index = np.random.choice(len(topic_probabilities), p=topic_probabilities)
    assigned_topic = likert_columns[assigned_topic_index]
    group_assignments[user_id] = assigned_topic

# Output CSVs for each group
for topic in likert_columns:
    topic_data = [(user_id, data.loc[data['User'] == user_id, 'Project Description'].values[0]) for user_id, assigned_topic in group_assignments.items() if assigned_topic == topic]
    pd.DataFrame(topic_data, columns=['User', 'Project Description']).to_csv(f"{topic}_group.csv", index=False)
    # Add each CSV into a list 
    list_csv_files.append(f"{topic}_group.csv")

# Count users in each project type
project_type_counts = {}
for assigned_topic in group_assignments.values():
    if assigned_topic in project_type_counts:
        project_type_counts[assigned_topic] += 1
    else:
        project_type_counts[assigned_topic] = 1

# Print the counts
for project_type, count in project_type_counts.items():
    print(f"{project_type}: {count} users")

# End Likert-----------------------------------------------------

# Read and clean dataset
warnings.filterwarnings("ignore")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    clean_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(clean_words)

def nlp_processing(file_name):
    data = pd.read_csv(file_name)
    data['Project Description'] = data['Project Description'].apply(remove_stopwords)
    texts = data['Project Description'].tolist()

    # Initialize LDATopic column to a default value
    data['LDATopic'] = -1

    # Fit BERTopic model
    bertopic_model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size=5)
    bertopic_topics, _ = bertopic_model.fit_transform(texts)
    data['BERTopic'] = bertopic_topics

    # Filter out outliers marked with -1 by BERTopic
    outlier_data = data[data['BERTopic'] == -1]
    outlier_texts = outlier_data['Project Description'].tolist()
    if not outlier_texts:
        # Sort the data by the topic
        sorted_data = data.sort_values(by='BERTopic')
        sorted_data.to_csv(f"NLP_{file_name}", index=False)

    else: 
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
        sorted_data.to_csv(f"NLP_{file_name}", index=False)
    return sorted_data

for file in list_csv_files:
    nlp_processing(file)
