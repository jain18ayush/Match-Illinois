import pandas as pd
import numpy as np

# Load the CSV data
data = pd.read_csv("data.csv")

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
