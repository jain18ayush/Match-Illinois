import pandas as pd 
import nltk
from nltk.corpus import stopwords
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

# Initialize BERTopic with a smaller min_topic_size
model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size=5, nr_topics='auto')

# Fit the model and get topics
topics, _ = model.fit_transform(data.Student_Responses)

# Assign the most probable topic to each document
data['Topic'] = topics

# Sort the data by the topic
data_sorted = data.sort_values(by='Topic')

# Save the sorted data with the topics to a new CSV file
data_sorted.to_csv('Bert_student_responses_with_topics_sorted.csv', index=False)

# Extract and display the topics
topic_overview = model.get_topic_info()
print(topic_overview.head(10))  # This will show the top topics, including outliers

# Output the topics and their descriptions to a CSV file
topic_overview.to_csv('Bert_topics_overview.csv', index=False)