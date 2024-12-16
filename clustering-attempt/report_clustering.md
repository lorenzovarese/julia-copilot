### Short Report on Clustering Approach for Julia Code

---

#### Approach

1. **Data Preparation**:
   - The dataset comprised multiple Julia projects stored in a JSON file.
   - Functions were filtered to include only those marked as `type: 'basic'` for clustering.

2. **Preprocessing**:
   - Julia-specific keywords and special characters were removed to reduce noise in the embeddings.
   - The functions were reconstructed into complete, cleaned versions by combining their signatures, bodies, and the `end` keyword, followed by a cleaning step to normalize the code.

3. **Embedding Generation**:
   - The cleaned functions were transformed into numerical embeddings using a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`).
   - Due to the large dataset (400,000 functions), embeddings were limited to a subset (10%) for faster processing.

4. **Clustering**:
   - K-means clustering was applied with a range of clusters (`k=2` to `k=20`).
   - Clustering quality was evaluated using:
     - **Inertia (Elbow Method)**: Measures within-cluster sum-of-squares.
     - **Silhouette Score**: Evaluates cohesion within clusters and separation between clusters.
     - **Calinski-Harabasz Index**: Measures the ratio of cluster dispersion to inter-cluster distances.

5. **Visualization**:
   - The metrics were plotted to evaluate the optimal number of clusters.

---

#### Results and Observations

- **Clustering Metrics**:
  - Both with and without removing Julia-specific keywords, the Elbow Method showed a steady decrease in inertia without a clear "elbow" point, suggesting no distinct clustering structure.
  - Silhouette Scores remained consistently low across all cluster counts, indicating poor clustering performance.
  - The Calinski-Harabasz Index similarly decreased monotonically, reflecting suboptimal clustering outcomes.

- **Conclusions**:
  - The approach of clustering based on embeddings did not yield meaningful clusters, as evident from the poor metric scores.
  - Removing Julia keywords and special characters did not improve results significantly.

---

#### Alternative Attempt: CrystalBLEU
A subsequent attempt to evaluate function similarity using the **CrystalBLEU** metric also failed to scale for the large dataset (400,000 functions), making it impractical.

