import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora, models
from bertopic import BERTopic
import warnings

# Function definitions and process flows
def nlp_processing_and_team_creation(file_name):
    # Read and clean dataset
    warnings.filterwarnings("ignore")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(text):
        words = text.split()
        clean_words = [word for word in words if word.lower() not in stop_words]
        return " ".join(clean_words)

    data = pd.read_csv(file_name)
    data['Project Description'] = data['Project Description'].apply(remove_stopwords)
    texts = data['Project Description'].tolist()

    # Initialize LDATopic column to a default value
    data['LDATopic'] = -1

    # Fit BERTopic model
    bertopic_model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size=5)
    bertopic_topics, _ = bertopic_model.fit_transform(texts)
    data['BERTopic'] = bertopic_topics

    # Process outliers with LDA
    outlier_data = data[data['BERTopic'] == -1]
    if not outlier_data.empty:
        outlier_texts = outlier_data['Project Description'].tolist()
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(outlier_texts)
        corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
        dictionary = corpora.Dictionary.from_corpus(corpus, id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
        outlier_data['LDATopic'] = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(outlier_texts))]
        data.update(outlier_data)

    # Group by 'LDATopic' and 'BERTopic' and create teams
    grouped = data.groupby(['LDATopic', 'BERTopic'])
    grouped_lists = [group['User'].tolist() for _, group in grouped]

    def create_teams(grouped_users):
        teams = []
        small_groups = []

        for group in grouped_users:
            while len(group) > 6:
                teams.append(group[:6])
                group = group[6:]
            if len(group) >= 3:
                teams.append(group)
            else:
                small_groups.extend(group)

        for member in small_groups:
            added_to_team = False
            for team in teams:
                if len(team) < 6:
                    team.append(member)
                    added_to_team = True
                    break
            if not added_to_team:
                teams.append([member])
                
        return teams

    final_teams = create_teams(grouped_lists)

    # Convert teams to DataFrame and save to CSV
    team_data = []
    for i, team in enumerate(final_teams):
        for user in team:
            user_data = data[data['User'] == user].iloc[0]
            team_data.append({'Team': i + 1, 'User': user, 'Project Description': user_data['Project Description'], 'LDATopic': user_data['LDATopic'], 'BERTopic': user_data['BERTopic']})
    team_df = pd.DataFrame(team_data)
    team_df.to_csv(f"Teams_{file_name}", index=False)
    return team_df

# Assuming 'data.csv' is the name of your data file
processed_files = []
for file in ["data.csv"]:  # Add more file names as required
    processed_files.append(nlp_processing_and_team_creation(file))

# 'processed_files' will hold the DataFrames for all processed files
