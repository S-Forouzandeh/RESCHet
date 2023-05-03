import requests
from bs4 import BeautifulSoup
import os
import dgl
import pandas as pd
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Model
from sklearn.linear_model import LinearRegression



# Get the current working directory
from tensorflow.python.keras.layers import embeddings
cwd = os.getcwd()
user_index_map={}
user_ratings={}
movie_index_map={}
recommended_movies = set()

# Open the "movies.csv" file in the "ml-latest-small" directory
filename = os.path.join(cwd, "Movilens-dataset", "ml-latest-small", "movies.csv")
if not os.path.exists(filename):
    # print(f"Error: '{filename}' does not exist.")
    exit()
with open(filename, "r+", encoding="utf-8") as f:

    # Read the "links.csv" file
    links_file = os.path.join(cwd, "Movilens-dataset", "ml-latest-small", "links.csv")
    if not os.path.exists(links_file):
        # print(f"Error: '{links_file}' does not exist.")
        exit()
    with open(links_file, "r", encoding="utf-8") as f_links:
        lines = f_links.readlines()[1:]  # skip the header line
        for line in lines:
            # Extract the movieId and imdbId
            movieId, imdbId, _ = line.strip().split(",")
            imdbId = "tt" + imdbId.zfill(7)  # format the imdbId

            # Scrape the director information from the IMDB website
            url = f"https://www. ... .com/title/{imdbId}/"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            director_tags = soup.select("div.credit_summary_item h4:-soup-contains('Director') + a")
            directors = [tag.text for tag in director_tags]

            # Update the "movies.csv" file with the director information
            with open(filename, "r+", encoding="utf-8") as f_movies:
                lines = f_movies.readlines()
                f_movies.seek(0)
                f_movies.truncate()
                for i, line in enumerate(lines):
                    if i == 0:
                        f_movies.write(line.strip() + ",director\n")  # add the "director" column header
                    else:
                        fields = line.strip().split(",")
                        if fields[0] == movieId:
                            f_movies.write(line.strip() + f",{','.join(directors)}\n")  # add the director information
                        else:
                            f_movies.write(line)  # write the existing row as is

# Created Heterogeneous Graph

# Read movies.csv into a Pandas DataFrame
movies_df = pd.read_csv("Movilens-dataset/ml-latest-small/movies.csv")

# Read links.csv into a Pandas DataFrame
links_df = pd.read_csv("Movilens-dataset/ml-latest-small/links.csv")

# Extract the movieIds and imdbIds from links_df
movieIds = links_df["movieId"].values
imdbIds = ["tt" + str(id).zfill(7) for id in links_df["imdbId"].values]

# Scrape the director information from the IMDB website for each movie
directors = []
for imdbId in imdbIds:
    url = f"https://www. ... .com/title/{imdbId}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    director_tags = soup.select("div.credit_summary_item h4:-soup-contains('Director') + a")
    director_names = [tag.text for tag in director_tags]
    directors.append(director_names)

# Create a DGL graph with three node types: user, movie, and director
graph = dgl.heterograph({
    ("user", "rates", "movie"): ([0], movieIds),
    ("user", "shares", "movie"): ([0], movieIds),
    ("movie", "has_director", "director"): (movieIds, range(len(movieIds), len(movieIds) + len(directors))),
})

# Add the nodes and their features to the graph
graph.nodes["user"].data["name"] = ["User1"]
graph.nodes["movie"].data["title"] = movies_df["title"].values
graph.nodes["director"].data["name"] = [director for director_list in directors for director in director_list]

# Print the graph's information
# print(graph)

# Open the "ratings.csv" file in the "ml-latest-small" directory
filename = os.path.join(cwd, "Movilens-dataset", "ml-latest-small", "ratings.csv")
if not os.path.exists(filename):
    # print(f"Error: '{filename}' does not exist.")
    exit()
with open(filename, "r", encoding="utf-8") as f:

    # Read the "ratings.csv" file and build the ground truth ratings matrix
    lines = f.readlines()[1:]  # skip the header line
    n_users = 0
    n_movies = 0
    for line in lines:
        userId, movieId, rating, _ = line.strip().split(",")
        userId = int(userId)
        movieId = int(movieId)
        rating = float(rating)
        if userId > n_users:
            n_users = userId
        if movieId > n_movies:
            n_movies = movieId
    ground_truth_ratings = np.zeros((n_users, n_movies))
    f.seek(0)
    lines = f.readlines()[1:]  # skip the header line
    for line in lines:
        userId, movieId, rating, _ = line.strip().split(",")
        userId = int(userId)
        movieId = int(movieId)
        rating = float(rating)
        ground_truth_ratings[userId-1, movieId-1] = rating
        target_matrix = np.zeros((n_users, n_movies))
        for user, ratings in ground_truth_ratings.iterrows():
            for movie in recommended_movies:
                if ratings[movie] > 0:
                    target_matrix[user_index_map[user], movie_index_map[movie]] = ratings[movie]

X=ground_truth_ratings
y=target_matrix
# Metapath2vec------------------------

# Load the movies dataset as a Pandas dataframe
movies_df = pd.read_csv("Movilens-dataset/ml-latest-small/movies.csv")

# Define the node types
node_types = {"user": ["userId"], "movie": ["movieId"], "director": ["director"]}

# Create a StellarGraph object from the Pandas dataframe
G = StellarGraph.from_pandas_edgelist(
    movies_df, edge_type="rated", source_column="userId", target_column="movieId", node_type=node_types
)

# Define the meta-paths
meta_paths = [("user", "rated", "movie"), ("movie", "directed_by", "director")]

# Create a node generator for GraphSAGE
generator = GraphSAGENodeGenerator(G, batch_size=50, num_samples=[10, 5])

# Specify the GraphSAGE model
model = GraphSAGE(layer_sizes=[32, 32], generator=generator)

# Train the model
train_gen = generator.flow(G.nodes(), targets=G.nodes())
model.fit(train_gen, epochs=20)

# Embed the nodes using Metapath2vec
metapath2vec = model.get_embedding_model(
    node_type="user",  # Start with user nodes
    meta_paths=meta_paths,
    embedding_dimension=128,
    negative_samples=5,
    walk_length=20,
    epochs=10,
)

# Generate random walks for user nodes
walks = G.random_walk(
    nodes=list(G.nodes()),  # Start with all nodes
    length=20,
    metapaths=meta_paths,
)

# Train a Word2Vec model on the random walks
model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, workers=4)

# Get the node embeddings for user nodes only
user_embeddings = {}
for node in G.nodes():
    if G.node_type(node) == "user":
        user_embeddings[node] = model.wv[node]

user_embeddings = model.embedding_vectors['user']
# print(user_embeddings)

#---- NonLinear Fusion -------------

# Define the number of meta-paths and embedding dimensions
num_paths = len(meta_paths)
embedding_dim = 128

# Define the inputs to the fusion layer
inputs = []
for i in range(num_paths):
    input_name = "path_{}_embedding".format(i)
    inputs.append(Input(shape=(embedding_dim,), name=input_name))

# Define the fusion layer
weights = Dense(units=num_paths, activation='softmax', name='weights')(inputs)
weighted_embeddings = []
for i in range(num_paths):
    embedding_weighted = Dense(units=embedding_dim, activation='sigmoid', name='embedding_weighted_{}'.format(i))(inputs[i])
    weighted_embeddings.append(embedding_weighted)
weighted_embeddings = np.array(weighted_embeddings)
weighted_embeddings = np.transpose(weighted_embeddings, [1, 0, 2])
weighted_sum = np.sum(np.multiply(weights, weighted_embeddings), axis=1)
output = Activation('sigmoid', name='output')(weighted_sum)

# Define the model
fusion_model = Model(inputs=inputs, outputs=output)

# Compile the model
fusion_model.compile(optimizer='adam', loss='binary_crossentropy')

# Get the embeddings for each meta-path
meta_path_embeddings = []
for i, path in enumerate(meta_paths):
    meta_path_embeddings.append(metapath2vec.predict(path))

# Compute the fusion vector for each node
fusion_vectors = {}
for node in G.nodes():
    if G.node_type(node) == "user":
        node_embeddings = []
        for i, path in enumerate(meta_paths):
            node_embeddings.append(meta_path_embeddings[i][node])
        node_fusion = fusion_model.predict(np.array(node_embeddings))
        fusion_vectors[node] = node_fusion

# ----------- Embedding Spectral Clustering -------

# Load the graph data
G = StellarGraph.from_pandas_edgelist(
    movies_df, edge_type="rated", source_column="userId", target_column="movieId", node_type=node_types
)

# Define the meta-paths
meta_paths = [("user", "rated", "movie"), ("movie", "directed_by", "director")]

# Create a node generator for GraphSAGE
generator = GraphSAGENodeGenerator(G, batch_size=50, num_samples=[10, 5])

# Specify the GraphSAGE model
model = GraphSAGE(layer_sizes=[32, 32], generator=generator)

# Train the model
train_gen = generator.flow(G.nodes(), targets=G.nodes())
model.fit(train_gen, epochs=20)

# Embed the nodes using Metapath2vec
embedding_model = model.get_embedding_model


# Obtain the node embeddings for user nodes only
for node in G.nodes():
    if G.node_type(node) == "user":
        user_embeddings[node] = embedding_model.wv[node]

# Create the adjacency matrix and degree matrix
A = G.to_adjacency_matrix()
D = np.diag(np.sum(A, axis=1))

# Compute the graph Laplacian matrix
L = D - A

# Compute the embedding similarity matrix
similarity_matrix = pairwise_distances(list(user_embeddings.values()), metric="cosine")

# Compute the eigenvectors of the Laplacian matrix
num_eigenvectors = len(user_embeddings)
eigenvalues, eigenvectors = eigsh(L, k=num_eigenvectors, which="SM")

# Select the number of clusters using the eigengap heuristic
sorted_indices = np.argsort(eigenvalues)
eigengaps = np.diff(np.sort(eigenvalues))
k = np.argmax(eigengaps) + 1

# Perform K-means clustering on the reduced vectors
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(eigenvectors[:, sorted_indices[:k]])

# Print the cluster labels for each user
for i, user_id in enumerate(user_embeddings.keys()):
 print(f"User {user_id} is in cluster {labels[i]}")

# Add cluster information to the graph
for i, node in enumerate(G.nodes()):
    if G.node_type(node) == "user":
        G.nodes[node]["cluster"] = labels[i]

# Define the meta-paths based on the clustering information
cluster_meta_paths = []
for i in range(k):
    cluster_meta_paths.append(
        ("user", f"belongs_to_{i}", "cluster"),
        (f"cluster_{i}", "belongs_to", "user"),
    )

# Create a node generator for the cluster-based meta-paths
cluster_generator = GraphSAGENodeGenerator(G, batch_size=50, num_samples=[10, 5])

# Specify the GraphSAGE model for the cluster-based meta-paths
cluster_model = GraphSAGE(layer_sizes=[32, 32], generator=cluster_generator)

# Train the model for the cluster-based meta-paths
cluster_train_gen = cluster_generator.flow(G.nodes(), targets=G.nodes())
cluster_model.fit(cluster_train_gen, epochs=20)

# Embed the nodes using the cluster-based meta-paths
cluster_embedding_model = cluster_model.get_embedding_model

#embeddings for user nodes only
user_embeddings_cluster = {}
for node in G.nodes():
 if G.node_type(node) == "user":
  user_embeddings_cluster[node] = cluster_embedding_model.wv[node]

#Define the group meta-path and atomic meta-paths
ug_meta_path = [("user", "in_group", "cluster"), ("cluster", "has_member", "user")]

# Define the atomic meta-paths based on the number of embedding spectral clusters
atomic_meta_paths = []
for i in range(k):
 atomic_meta_path = [("user", f"in_{i}", f"cluster_{i}"), (f"cluster_{i}", f"has_member", "user")]
 atomic_meta_paths.append(atomic_meta_path)

 #Compute the similarity matrix using the group meta-path
similarity_matrix_ug = pairwise_distances(list(user_embeddings_cluster.values()), metric="cosine",
X_norm=np.array(list(user_embeddings_cluster.values())))

# Compute the similarity matrix using the atomic meta-paths
similarity_matrices_atomic = []
for atomic_meta_path in atomic_meta_paths:
 embedding_model_atomic = cluster_model.get_embedding_model
user_embeddings_atomic = {}
for node in G.nodes():
 if G.node_type(node) == "user":
  user_embeddings_atomic[node] = embedding_model_atomic.wv[node]
similarity_matrix_atomic = pairwise_distances(list(user_embeddings_atomic.values()), metric="cosine",
X_norm=np.array(list(user_embeddings_atomic.values())))
similarity_matrices_atomic.append(similarity_matrix_atomic)

# Compute the weighted similarity matrix
weighted_similarity_matrix = np.zeros_like(similarity_matrix_ug)
for i in range(k):
 weighted_similarity_matrix += similarity_matrices_atomic[i] * similarity_matrix_ug[:, i].reshape(-1, 1)

 # Perform K-means clustering on the weighted similarity matrix
 kmeans_weighted = KMeans(n_clusters=k, random_state=42)
 labels_weighted = kmeans_weighted.fit_predict(weighted_similarity_matrix)

 #Evaluate the clustering performance
 labels_true = {}
 with open(links_file, "r", encoding="utf-8") as f_links:
     for line in f_links:
         node_id, label = line.strip().split(",")
         if G.node_type(node_id) == "user":
             labels_true[node_id] = int(label)

 ari_weighted = adjusted_rand_score(labels_true, labels_weighted)
 # print(f"Weighted meta-path clustering ARI: {ari_weighted}")

# Hadamard product----------------

# Compute the similarity matrix using the submeta-paths
similarity_matrices = []
for submeta_path in user_embeddings_atomic:
    embedding_model = model.get_embedding_model
    for node in G.nodes():
        if G.node_type(node) == "user":
            user_embeddings[node] = embedding_model.wv[node]
    similarity_matrix = pairwise_distances(
        list(user_embeddings.values()), metric="cosine",
        X_norm=np.array(list(user_embeddings.values()))
    )
    similarity_matrices.append(similarity_matrix)

# Compute the Hadamard product of embedded vectors from different submeta-paths
weighted_similarity_matrix = np.ones_like(similarity_matrices[0])
for similarity_matrix in similarity_matrices:
    weighted_similarity_matrix *= similarity_matrix

#-------- Recommendation --------

# Get the embeddings for all users
for node in G.nodes():
    if G.node_type(node) == "user":
        user_embeddings[node] = embedding_model.wv[node]

# Calculate the similarity matrix
similarity_matrix = cosine_similarity(list(user_embeddings.values()))

# Example of recommending movies for a user with ID 'u1'
target_user = 'u1'
# Get the indices of the top 5 most similar users
similar_users_indices = similarity_matrix[user_index_map[target_user]].argsort()[::-1][1:6]
# Get the IDs of the top 5 most similar users
similar_users = [list(user_embeddings.keys())[i] for i in similar_users_indices]
# Get the movies that the top 5 most similar users have rated highly
recommended_movies = set()
for user in similar_users:
    recommended_movies.update(user_ratings[user].nlargest(5).index)
# Remove the movies that the target user has already rated
recommended_movies = recommended_movies - set(user_ratings[target_user].index)
# Print the recommended movies
# print(recommended_movies)

# ------------ Evaluation Method------------------------

train_sizes = [0.2, 0.4, 0.6, 0.8]
# Create an empty list to store the results
results = []

for size in train_sizes:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the ratings for the testing set
    y_pred = model.predict(X_test)

    # Calculate the MAE and RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Calculate the improvement compared to the previous train size (if any)
    if results:
        prev_mae, prev_rmse = results[-1][1:]
        mae_improvement = (prev_mae - mae) / prev_mae * 100
        rmse_improvement = (prev_rmse - rmse) / prev_rmse * 100
    else:
        mae_improvement = np.nan
        rmse_improvement = np.nan

    # Add the results to the list
    results.append((size, mae, rmse, mae_improvement, rmse_improvement))

# Print the results
# print("Train Size\tMAE\t\tRMSE\t\tMAE Improvement\tRMSE Improvement")
for size, mae, rmse, mae_improvement, rmse_improvement in results:
    print(f"{size:.0%}\t\t{mae:.4f}\t\t{rmse:.4f}\t\t{mae_improvement:.2f}%\t\t{rmse_improvement:.2f}%")


