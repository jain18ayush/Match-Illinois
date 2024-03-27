
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import gensim
from bertopic import BERTopic
import warnings

# Load and prepare the dataset
data = pd.read_csv("data.csv")
list_csv_files = []

# Likert scale processing and initial group assignment
likert_columns = ["Web Dev", "ML", "Mobile Dev", "App Dev", "Data Analytics"]
weights = {1: 0.5, 2: 0.4, 3: 0.05, 4: 0.03, 5: 0.02}
probabilities = data[likert_columns].apply(lambda x: [weights[val] for val in x], axis=1)
group_assignments = {}

for idx, topic_probabilities in probabilities.items():
    user_id = data.loc[idx, 'User']
    assigned_topic_index = np.random.choice(len(topic_probabilities), p=topic_probabilities)
    assigned_topic = likert_columns[assigned_topic_index]
    group_assignments[user_id] = assigned_topic

for topic in likert_columns:
    topic_data = [(user_id, data.loc[data['User'] == user_id, 'Project Description'].values[0])
                  for user_id, assigned_topic in group_assignments.items() if assigned_topic == topic]
    group_csv = f"{topic}_group.csv"
    pd.DataFrame(topic_data, columns=['User', 'Project Description']).to_csv(group_csv, index=False)
    list_csv_files.append(group_csv)

# NLP processing with BERTopic and LDA
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    return " ".join(word for word in words if word.lower() not in stop_words)

def nlp_processing(file_name):
    data = pd.read_csv(file_name)
    data['Project Description'] = data['Project Description'].apply(remove_stopwords)
    texts = data['Project Description'].tolist()

    bertopic_model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size=5)
    bertopic_topics, _ = bertopic_model.fit_transform(texts)
    data['BERTopic'] = bertopic_topics

    # Additional processing for outliers and LDA...

    sorted_data = data.sort_values(by='BERTopic')
    sorted_data.to_csv(f"NLP_{file_name}", index=False)
    return sorted_data

nlp_csv_files = []
for file in list_csv_files:
    processed_data = nlp_processing(file)
    nlp_csv_files.append(f"NLP_{file}")

# Team formation and final CSV compilation
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

final_data = []
with open("Combined_Teams.csv", "w") as outfile:
    for file in nlp_csv_files:
        # Extract the group name from the filename
        group_name = file.split('_')[1]  # This gets the second word in the filename

        data = pd.read_csv(file)
        grouped_lists = data.groupby('BERTopic').apply(lambda x: x['User'].tolist()).tolist()
        final_teams = create_teams(grouped_lists)

        # Write the group name as a header for the section
        outfile.write(f"{group_name}\n")

        team_data = []
        for i, team in enumerate(final_teams):
            for user in team:
                user_data = data[data['User'] == user].iloc[0]
                team_data.append({
                    'Team': i + 1,
                    'User': user,
                    'Project Description': user_data['Project Description'],
                    'LDATopic': user_data.get('LDATopic', -1),
                    'BERTopic': user_data['BERTopic']
                })
        
        # Convert the team data to a DataFrame and then to CSV, excluding the index and header
        team_df = pd.DataFrame(team_data)
        team_df.to_csv(outfile, index=False, header=False)

        # Write a newline to separate each group's section
        outfile.write("\n")
