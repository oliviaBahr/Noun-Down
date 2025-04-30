import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence


def create_directory(dir_name: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory: {dir_name}")
    else:
        print(f"Directory already exists: {dir_name}")


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data into pandas DataFrame."""
    df = pd.read_csv(file_path)
    print(f"Dataframe shape: {df.shape}")
    print(f"First few rows:\n{df[['noun', 'total']].head()}")
    return df


def perform_elbow_analysis(
    data_scaled: np.ndarray, k_range: Sequence[int], output_dir: str
) -> List[float]:
    """Perform elbow method analysis to find optimal k."""
    inertia: List[float] = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # Plot the elbow method results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, "o-", markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method for Optimal k", fontsize=14)
    plt.grid(True)
    plt.xticks(k_range)
    plt.savefig(os.path.join(output_dir, "elbow_method.png"))
    plt.close()

    return inertia


def run_kmeans(data_scaled: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]:
    """Run K-means clustering with the specified k."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    return kmeans, clusters


def create_cluster_visualization(
    df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler, k: int, output_dir: str
) -> None:
    """Create and save visualization of clusters."""
    # Sort by total value to create a more meaningful visualization
    df_sorted = df.sort_values("total")
    df_sorted["index"] = range(len(df_sorted))

    plt.figure(figsize=(12, 8))

    # Plot each cluster with a different color
    for cluster_id in range(k):
        cluster_data = df_sorted[df_sorted["cluster"] == cluster_id]
        plt.scatter(
            cluster_data["index"],
            cluster_data["total"],
            label=f"Cluster {cluster_id}",
            alpha=0.8,
            s=50,
        )

    # Add cluster centroids
    for i, centroid in enumerate(scaler.inverse_transform(kmeans.cluster_centers_)):
        plt.axhline(y=centroid[0], color=f"C{i}", linestyle="--", alpha=0.5)

    plt.xlabel("Index (sorted by total value)", fontsize=12)
    plt.ylabel("Total Value", fontsize=12)
    plt.title("K-means Clustering of Nouns by Total Value", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add a few noun labels for context (select some from each cluster)
    for cluster_id in range(k):
        cluster_data = df_sorted[df_sorted["cluster"] == cluster_id]
        # Get some sample nouns from each cluster (first, middle, last)
        if len(cluster_data) > 0:
            samples = [0, len(cluster_data) // 2, len(cluster_data) - 1]
            for idx in samples:
                if idx < len(cluster_data):
                    sample = cluster_data.iloc[idx]
                    plt.annotate(
                        sample["noun"],
                        (sample["index"], sample["total"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmeans_clustering.png"))
    plt.close()


def print_cluster_statistics(df: pd.DataFrame, k: int) -> None:
    """Print statistics for each cluster."""
    print("\nCluster Statistics:")
    for cluster_id in range(k):
        cluster_data = df[df["cluster"] == cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  Number of nouns: {len(cluster_data)}")
        print(f"  Average total value: {cluster_data['total'].mean():.4f}")
        print(f"  Min total value: {cluster_data['total'].min():.4f}")
        print(f"  Max total value: {cluster_data['total'].max():.4f}")
        print(
            f"  Sample nouns: {', '.join(cluster_data['noun'].sample(min(5, len(cluster_data))).tolist())}"
        )


def main() -> None:
    """Main function to run the analysis."""
    # Create output directory for visualizations
    output_dir: str = "noun_cluster_visualizations"
    create_directory(output_dir)

    # Load the CSV file
    df: pd.DataFrame = load_data("results.csv")

    # Extract only 'total' column for clustering
    data: np.ndarray = df[["total"]].values

    # Standardize the data
    scaler: StandardScaler = StandardScaler()
    data_scaled: np.ndarray = scaler.fit_transform(data)

    # Determine optimal number of clusters using elbow method
    k_range: range = range(1, 10)
    inertia: List[float] = perform_elbow_analysis(data_scaled, k_range, output_dir)

    # Choose k=3 clusters (or adjust based on elbow method results)
    k: int = 3
    kmeans, clusters = run_kmeans(data_scaled, k)

    # Add cluster information to the original dataframe
    df["cluster"] = clusters

    # Create visualization
    create_cluster_visualization(df, kmeans, scaler, k, output_dir)

    # Print statistics for each cluster
    print_cluster_statistics(df, k)

    # Generate additional visualization: Histogram of total values by cluster
    plt.figure(figsize=(12, 8))
    for cluster_id in range(k):
        cluster_data = df[df["cluster"] == cluster_id]
        plt.hist(
            cluster_data["total"], bins=30, alpha=0.5, label=f"Cluster {cluster_id}"
        )

    plt.xlabel("Total Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Total Values by Cluster", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "cluster_histograms.png"))
    plt.close()

    print(f"\nVisualizations saved in directory: {output_dir}")


if __name__ == "__main__":
    main()
