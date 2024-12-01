import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
import random
from pandarallel import pandarallel
import os

# Initialize pandarallel
pandarallel.initialize()

# Julia-specific keywords and special characters to remove
JULIA_KEYWORDS = {
    "baremodule", "begin", "break", "catch", "const", "continue", "do", "else", "elseif", "end",
    "export", "false", "finally", "for", "function", "global", "if", "import", "let", "local",
    "macro", "module", "quote", "return", "struct", "true", "try", "using", "while"
}
SPECIAL_CHARACTERS = r"[()\[\]{}<>:;.,=+\-*\/\\%&|^~!?]"

def load_dataset(file_path):
    """Loads the dataset from a JSON file."""
    print("Loading dataset...")
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"Dataset loaded. Total projects: {len(data)}")
    return data


def filter_basic_functions(julia_code):
    """Filters functions with 'type': 'basic'."""
    print("Filtering basic functions...")
    basic_functions = [
        func
        for item in julia_code
        for func in item["functions"]
        if func["type"] == "basic"
    ]
    print(f"Found {len(basic_functions)} basic functions.")
    return basic_functions


def reconstruct_functions(functions):
    """Reconstructs Julia functions by concatenating signature, body, and `end`."""
    print("Reconstructing functions...")
    for func in functions:
        func["full_code"] = f"{func['signature']}\n{func['body']}\nend"
    print(f"Reconstructed {len(functions)} functions.")
    return functions

def clean_function_code(function_code):
    """
    Cleans Julia function code by removing keywords, special characters, and extra whitespace.
    """
    # Remove Julia keywords
    pattern = r"\b(" + "|".join(re.escape(keyword) for keyword in JULIA_KEYWORDS) + r")\b"
    cleaned_code = re.sub(pattern, "", function_code)
    
    # Remove special characters
    cleaned_code = re.sub(SPECIAL_CHARACTERS, " ", cleaned_code)
    
    # Normalize whitespace
    cleaned_code = re.sub(r"\s+", " ", cleaned_code).strip()
    
    return cleaned_code


def reconstruct_functions(functions):
    """Reconstructs and cleans Julia functions."""
    print("Reconstructing and cleaning functions...")
    for func in functions:
        raw_code = f"{func['signature']}\n{func['body']}\nend"
        func["full_code"] = clean_function_code(raw_code)
    print(f"Cleaned and reconstructed {len(functions)} functions.")
    return functions


# The rest of your code remains the same, with minor updates to use cleaned code for embedding generation.

def extract_embeddings(functions, model_name='all-MiniLM-L6-v2', batch_size=100, limit_ratio=1.0, embedding_file="embeddings.npy"):
    """
    Generates embeddings for the cleaned function bodies in parallel.
    """
    if os.path.exists(embedding_file):
        print("Loading saved embeddings...")
        embeddings = np.load(embedding_file)
        print(f"Loaded embeddings from {embedding_file}.")
        return embeddings

    print("Converting functions to DataFrame for parallel processing...")
    df = pd.DataFrame(functions)
    
    print("Extracting cleaned function code for embedding...")
    code_snippets = df["full_code"].tolist()
    if limit_ratio < 1.0:
        sample_size = int(limit_ratio * len(code_snippets))
        print(f"Limiting embeddings to {sample_size} functions ({limit_ratio * 100:.1f}%).")
        code_snippets = random.sample(code_snippets, sample_size)

    print("Loading embedding model...")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings in batches...")
    embeddings = []
    for i in range(0, len(code_snippets), batch_size):
        batch = code_snippets[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(code_snippets) - 1) // batch_size + 1}...")
        batch_embeddings = model.encode(batch)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)  # Combine all batches
    np.save(embedding_file, embeddings)
    print(f"Generated embeddings for {len(code_snippets)} functions. Saved to {embedding_file}.")
    return embeddings


def evaluate_clustering(embeddings, max_clusters):
    """Evaluates clustering metrics for a range of cluster counts."""
    print("Evaluating clustering metrics...")
    metrics = []

    for num_clusters in range(2, max_clusters + 1, 2):
        print(f"Processing {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(embeddings)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(embeddings, clusters)
        calinski_harabasz = calinski_harabasz_score(embeddings, clusters)

        metrics.append((num_clusters, inertia, silhouette, calinski_harabasz, kmeans))
        print(f"Clusters: {num_clusters} | Inertia: {inertia:.2f} | Silhouette: {silhouette:.2f} | Calinski-Harabasz: {calinski_harabasz:.2f}")

    return metrics


def plot_metrics(metrics, output_file):
    """Plots clustering metrics and saves the plot."""
    print("Plotting metrics...")
    num_clusters, inertia, silhouette, calinski_harabasz, _ = zip(*metrics)

    plt.figure(figsize=(15, 5))

    # Elbow Method
    plt.subplot(1, 3, 1)
    plt.plot(num_clusters, inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    # Silhouette Score
    plt.subplot(1, 3, 2)
    plt.plot(num_clusters, silhouette, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")

    # Calinski-Harabasz Index
    plt.subplot(1, 3, 3)
    plt.plot(num_clusters, calinski_harabasz, marker="o")
    plt.title("Calinski-Harabasz Index")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Metrics plot saved to {output_file}")


def main():
    # Parameters
    dataset_path = "data/combined_projects.json"
    max_clusters = 20
    plot_file = "clustering_metrics.png"
    embedding_file = "embeddings.npy"
    limit_ratio = 0.1  # Limit embeddings to 10%

    # Load dataset
    julia_code = load_dataset(dataset_path)

    # Filter and preprocess functions
    basic_functions = filter_basic_functions(julia_code)
    basic_functions = reconstruct_functions(basic_functions)

    # Generate embeddings
    embeddings = extract_embeddings(basic_functions, limit_ratio=limit_ratio, embedding_file=embedding_file)

    # Evaluate clustering
    metrics = evaluate_clustering(embeddings, max_clusters)

    # Plot metrics
    plot_metrics(metrics, plot_file)


if __name__ == "__main__":
    main()
