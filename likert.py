import pandas as pd
from sklearn.cluster import KMeans

# Read the CSV file into a DataFrame
df_real_data = pd.read_csv('/Users/nicolehu/Documents/GitHub/Match-Illinois/Main Likert and Project Description Data - Sheet1 (1).csv')

# Apply K-Means clustering to group the people into 5 groups (one for each category)
num_groups = 5
df_real_data_cluster = df_real_data.drop(columns=['Project Description'], axis=1)

# Apply K-means clustering
kmeans_real = KMeans(n_clusters=num_groups, random_state=42)
kmeans_real.fit(df_real_data_cluster.iloc[:, 1:])

# Get cluster labels
cluster_labels_real = kmeans_real.labels_

# Define category names
category_names = ['Web Dev', 'ML', 'Mobile Dev', 'App Dev', 'Data Analytics']

# Initialize a dictionary to hold the data for each category
category_data = {name: [] for name in category_names}

# Populate the dictionary with user IDs and project descriptions for each category based on cluster labels
for i, label in enumerate(cluster_labels_real):
    category = category_names[label]
    user_id = df_real_data.loc[i, 'User']
    project_description = df_real_data.loc[i, 'Project Description']
    category_data[category].append({'User': user_id, 'Project Description': project_description})

# Create a CSV file for each category with the users and their project descriptions
for category, data in category_data.items():
    # Create a DataFrame for the current category
    category_df = pd.DataFrame(data)
    # Generate the CSV file name
    csv_filename = f"{category.replace('/', '_').replace(' ', '_')}_group.csv"
    # Define the path to save the file (here it's specified as '/mnt/data/', which is the sandbox environment in this context)
    output_path = f'/Users/nicolehu/Documents/GitHub/Match-Illinois/{csv_filename}'
    # Save the DataFrame to a CSV file
    category_df.to_csv(output_path, index=False)

