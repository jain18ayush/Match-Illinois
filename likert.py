import pandas as pd
from sklearn.cluster import KMeans

# Read the CSV file into a DataFrame
df_real_data = pd.read_csv('/Users/nicolehu/Documents/GitHub/Match-Illinois/Main Likert and Project Description Data - Sheet1 (1).csv')

# Drop the 'Project Description' column as it's not needed for K-Means clustering
df_real_data_cluster = df_real_data.drop('Project Description', axis=1)

# Apply K-Means clustering to group the people into 5 groups (one for each category)
num_groups = 5

# Apply K-means clustering
kmeans_real = KMeans(n_clusters=num_groups, random_state=42)
kmeans_real.fit(df_real_data_cluster.iloc[:, 1:])

# Get cluster labels
cluster_labels_real = kmeans_real.labels_

# Define category names
category_names = ['Web Dev', 'ML', 'Mobile Dev', 'App Dev', 'Data Analytics']

# Group individuals by their cluster labels (category)
groups_real = {category_names[i]: [] for i in range(num_groups)}
for i, label in enumerate(cluster_labels_real):
    groups_real[category_names[label]].append(i + 1)  # Adding 1 to start indexing from 1

# Create a CSV file for each category with the users listed
for category, members in groups_real.items():
    # Create a DataFrame for the current category with users
    members_df = pd.DataFrame(members, columns=['User'])
    # Generate the CSV file name
    csv_filename = f"{category.replace('/', '_').replace(' ', '_')}_group.csv"
    # Save the DataFrame to a CSV file
    members_df.to_csv(f'/Users/nicolehu/Documents/GitHub/Match-Illinois/{csv_filename}', index=False)
