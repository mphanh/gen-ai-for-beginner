import os
import pandas as pd
import numpy as np
import openai
from azure.cosmos import CosmosClient
from sklearn.neighbors import NearestNeighbors

# Creating our Knowledge base
# Initialize Cosmos Client
url = os.getenv('COSMOS_DB_ENDPOINT')
key = os.getenv('COSMOS_DB_KEY')
client = CosmosClient(url, credential=key)

# Select database
database_name = 'rag-cosmos-db'
database = client.get_database_client(database_name)

# Select container
container_name = 'data'
container = database.get_container_client(container_name)

# Initialize an empty DataFrame
df = pd.DataFrame(columns=['path', 'text'])


# splitting our data into chunks
data_paths= ["data/frameworks.md?WT.mc_id=academic-105485-koreyst", "data/own_framework.md?WT.mc_id=academic-105485-koreyst", "data/perceptron.md?WT.mc_id=academic-105485-koreyst"]

for path in data_paths:
    with open(path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Append the file path and text to the DataFrame
    df = df.append({'path': path, 'text': file_content}, ignore_index=True)

df.head()


# SPLIT DATA TO CHUNKS
def split_text(text, max_length, min_length):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) < max_length and len(' '.join(current_chunk)) > min_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # If the last chunk didn't reach the minimum length, add it anyway
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Assuming analyzed_df is a pandas DataFrame and 'output_content' is a column in that DataFrame
splitted_df = df.copy()
splitted_df['chunks'] = splitted_df['text'].apply(lambda x: split_text(x, 400, 300))

# Assuming 'chunks' is a column of lists in the DataFrame splitted_df, we will split the chunks into different rows
flattened_df = splitted_df.explode('chunks')

flattened_df.head()


# TEXT TO EMBEDDINGS
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY") 
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
openai.api_version = "2023-07-01-preview"

client = OpenAI(api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"))
def create_embeddings(text, model="text-embedding-ada-002-2"):
    # Create embeddings for each document chunk
    embeddings = openai.embeddings.create(input = text, model=model).data[0].embedding
    return embeddings

#embeddings for the first chunk
create_embeddings(flattened_df['chunks'][0])

# create embeddings for the whole data chunks and store them in a list

embeddings = []
for chunk in flattened_df['chunks']:
    embeddings.append(create_embeddings(chunk))

# store the embeddings in the dataframe
flattened_df['embeddings'] = embeddings

flattened_df.head()


# SEARCH INDEX AND RERANKING
embeddings = flattened_df['embeddings'].to_list()

# Create the search index
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embeddings)

# To query the index, you can use the kneighbors method
distances, indices = nbrs.kneighbors(embeddings)

# Store the indices and distances in the DataFrame
flattened_df['indices'] = indices.tolist()
flattened_df['distances'] = distances.tolist()

flattened_df.head()

# Your text question
question = "what is a perceptron?"

# Convert the question to a query vector
query_vector = create_embeddings(question)  # You need to define this function

# Find the most similar documents
distances, indices = nbrs.kneighbors([query_vector])

index = []
# Print the most similar documents
for i in range(3):
    index = indices[0][i]
    for index in indices[0]:
        print(flattened_df['chunks'].iloc[index])
        print(flattened_df['path'].iloc[index])
        print(flattened_df['distances'].iloc[index])
    else:
        print(f"Index {index} not found in DataFrame")