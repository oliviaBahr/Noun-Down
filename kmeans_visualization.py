import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import sys
from typing import List, Dict, Tuple


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load results.csv file using pandas.

    Args:
        file_path: Path to the results.csv file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(file_path)


def display_dataframe_info(df: pd.DataFrame) -> None:
    """
    Display information about the DataFrame.

    Args:
        df: Pandas DataFrame to display information about.
    """
    print("\nDataframe info:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(f"\nNumeric columns that will be used for clustering: {numeric_cols}")

    print("\nFirst 5 rows:")
    print(df.head(5))


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess data for K-means clustering.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of preprocessed DataFrame and list of numeric column names.
    """
    # Get numeric columns only (excluding 'noun' if it's numeric)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "noun" in numeric_cols:
        numeric_cols.remove("noun")

    # Remove rows with missing values in numeric columns
    df_clean = df.dropna(subset=numeric_cols)

    return df_clean, numeric_cols


def determine_optimal_k(data: np.ndarray, max_k: int = 10) -> Tuple[plt.Figure, int]:
    """
    Determine the optimal number of clusters using the Elbow method and Silhouette score.

    Args:
        data: Numpy array of scaled data.
        max_k: Maximum number of clusters to try.

    Returns:
        Tuple of matplotlib figure and optimal k value.
    """
    # Limit max_k based on data size
    max_k = min(max_k, len(data) - 1)

    # Calculate metrics for different k values
    inertia_values = []
    silhouette_values = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

        # Calculate silhouette score
        if len(np.unique(kmeans.labels_)) > 1:  # Ensure we have more than one cluster
            silhouette_values.append(silhouette_score(data, kmeans.labels_))
        else:
            silhouette_values.append(0)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Elbow Method
    ax1.plot(list(k_values), inertia_values, "bo-")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Sum of Squared Distances)")
    ax1.set_title("Elbow Method for Optimal k")
    ax1.grid(True)

    # Plot Silhouette Scores
    ax2.plot(list(k_values), silhouette_values, "ro-")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Method for Optimal k")
    ax2.grid(True)

    plt.tight_layout()

    # Determine optimal k using silhouette score
    optimal_k = k_values[np.argmax(silhouette_values)]

    return fig, optimal_k


def perform_kmeans(data: np.ndarray, k: int) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-means clustering.

    Args:
        data: Numpy array of scaled data.
        k: Number of clusters.

    Returns:
        Tuple of cluster labels and fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


def create_pca_visualization(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    center_points: np.ndarray,
) -> plt.Figure:
    """
    Create PCA visualization of clusters.

    Args:
        data: Numpy array of scaled data.
        labels: Cluster labels.
        feature_names: Names of features used for clustering.
        center_points: Cluster centers.

    Returns:
        Matplotlib figure.
    """
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    # Transform cluster centers
    centers_pca = pca.transform(center_points)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot
    scatter = ax.scatter(
        pca_result[:, 0], pca_result[:, 1], c=labels, cmap="viridis", alpha=0.6, s=50
    )

    # Plot cluster centers
    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        marker="X",
        s=200,
        c="red",
        edgecolor="black",
        linewidth=2,
        label="Cluster Centers",
    )

    # Add labels and title
    ax.set_xlabel(
        f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
    )
    ax.set_ylabel(
        f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
    )
    ax.set_title("PCA of K-means Clustering Results", fontsize=15)

    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.legend(loc="upper right")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add text explaining the PCA
    feature_contributions = pd.DataFrame(
        pca.components_, columns=feature_names, index=["PC1", "PC2"]
    ).T

    # Finding top contributors to each PC
    top_features_pc1 = feature_contributions["PC1"].abs().nlargest(3).index.tolist()
    top_features_pc2 = feature_contributions["PC2"].abs().nlargest(3).index.tolist()

    text = f"Top features in PC1: {', '.join(top_features_pc1)}\n"
    text += f"Top features in PC2: {', '.join(top_features_pc2)}"

    plt.figtext(
        0.5,
        0.01,
        text,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig


def create_feature_importance_visualization(
    kmeans_model: KMeans, feature_names: List[str], k: int
) -> plt.Figure:
    """
    Create visualizations of cluster centers to show feature importance.

    Args:
        kmeans_model: Fitted KMeans model.
        feature_names: Names of features used for clustering.
        k: Number of clusters.

    Returns:
        Matplotlib figure.
    """
    centers = kmeans_model.cluster_centers_

    # Creating a DataFrame for the centers
    centers_df = pd.DataFrame(centers, columns=feature_names)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(feature_names)), 8))

    # Plot heatmap
    sns.heatmap(
        centers_df,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Feature Value (Scaled)"},
    )

    # Add labels and title
    ax.set_title("Cluster Centers Heatmap", fontsize=15)
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)

    plt.tight_layout()

    return fig


def create_parallel_coordinates(
    df: pd.DataFrame, feature_names: List[str], labels: np.ndarray
) -> plt.Figure:
    """
    Create parallel coordinates plot for visualizing clusters across features.

    Args:
        df: Original DataFrame.
        feature_names: Names of features used for clustering.
        labels: Cluster labels.

    Returns:
        Matplotlib figure.
    """
    # Create copy of DataFrame with only the features used for clustering
    df_features = df[feature_names].copy()

    # Add cluster labels
    df_features["Cluster"] = labels

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(feature_names)), 8))

    # Create parallel coordinates plot
    pd.plotting.parallel_coordinates(
        df_features, "Cluster", color=plt.cm.viridis.colors, alpha=0.5
    )

    # Add labels and title
    ax.set_title("Parallel Coordinates Plot of Clusters", fontsize=15)

    # Adjust labels
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig


def create_cluster_distribution(
    df: pd.DataFrame, labels: np.ndarray
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create visualization showing the distribution of clusters.

    Args:
        df: Original DataFrame.
        labels: Cluster labels.

    Returns:
        Tuple of matplotlib figure and DataFrame with cluster statistics.
    """
    # Add cluster labels to the original DataFrame
    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = labels

    # Count samples in each cluster
    cluster_counts = df_with_clusters["Cluster"].value_counts().sort_index()

    # Calculate percentages
    total_samples = len(df_with_clusters)
    cluster_percentages = (cluster_counts / total_samples * 100).round(2)

    # Create statistics DataFrame
    cluster_stats = pd.DataFrame(
        {"Count": cluster_counts, "Percentage": cluster_percentages}
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    bars = ax.bar(
        cluster_stats.index,
        cluster_stats["Count"],
        color=plt.cm.viridis(np.linspace(0, 1, len(cluster_stats))),
    )

    # Add labels and title
    ax.set_title("Distribution of Clusters", fontsize=15)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)

    # Add count and percentage labels on top of bars
    for bar, count, percentage in zip(
        bars, cluster_stats["Count"], cluster_stats["Percentage"]
    ):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{count} ({percentage}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    return fig, cluster_stats


def create_cluster_profiles(
    df: pd.DataFrame, feature_names: List[str], labels: np.ndarray, k: int
) -> plt.Figure:
    """
    Create radar charts showing cluster profiles.

    Args:
        df: Original DataFrame.
        feature_names: Names of features used for clustering.
        labels: Cluster labels.
        k: Number of clusters.

    Returns:
        Matplotlib figure.
    """
    # Add cluster labels to the original DataFrame
    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = labels

    # Calculate mean values for each feature within each cluster
    cluster_profiles = df_with_clusters.groupby("Cluster")[feature_names].mean()

    # Normalize the values for radar chart
    scaler = StandardScaler()
    cluster_profiles_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_profiles),
        index=cluster_profiles.index,
        columns=cluster_profiles.columns,
    )

    # Set up the radar chart
    # Calculate the angle for each feature
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create figure
    fig, axes = plt.subplots(1, k, figsize=(k * 5, 5), subplot_kw=dict(polar=True))

    # Ensure axes is array-like even for single cluster
    if k == 1:
        axes = [axes]

    # Plot each cluster profile
    for i, ax in enumerate(axes):
        if i < len(cluster_profiles_scaled):
            values = cluster_profiles_scaled.iloc[i].values.tolist()
            values += values[:1]  # Close the loop

            # Plot the values
            ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {i}")
            ax.fill(angles, values, alpha=0.25)

            # Set labels
            ax.set_thetagrids(np.degrees(angles[:-1]), feature_names)

            # Set title
            ax.set_title(f"Cluster {i} Profile", y=1.1)

            # Set y ticks
            ax.set_ylim(-3, 3)
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.set_yticklabels(["Very Low", "Low", "Average", "High", "Very High"])

    plt.tight_layout()

    return fig


def save_visualization(fig: plt.Figure, filename: str) -> str:
    """
    Save the visualization as a png file.

    Args:
        fig: Matplotlib figure to save.
        filename: Name of the file.

    Returns:
        Path to the saved file.
    """
    # Create output directory if it doesn't exist
    output_dir = "kmeans_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Generate full filepath
    filepath = f"{output_dir}/{filename}"

    # Save the figure
    fig.savefig(filepath, dpi=300, bbox_inches="tight")

    return filepath


def generate_kmeans_visualizations(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate and save K-means clustering visualizations.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary of visualization types and their file paths.
    """
    saved_files = {}

    # Preprocess data
    print("Preprocessing data...")
    df_clean, numeric_cols = preprocess_data(df)

    if len(numeric_cols) < 2:
        print("Error: Need at least 2 numeric columns for clustering.")
        return saved_files

    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean[numeric_cols])

    # Determine optimal number of clusters
    print("Determining optimal number of clusters...")
    max_k = min(10, len(df_clean) // 5, 15)  # Limit max_k
    optimal_k_fig, optimal_k = determine_optimal_k(scaled_data, max_k)

    # Save optimal k visualization
    optimal_k_path = save_visualization(optimal_k_fig, "optimal_k_determination.png")
    saved_files["optimal_k"] = optimal_k_path
    plt.close(optimal_k_fig)

    print(f"Optimal number of clusters (k): {optimal_k}")

    # Perform K-means clustering
    print(f"Performing K-means clustering with k={optimal_k}...")
    labels, kmeans_model = perform_kmeans(scaled_data, optimal_k)

    # Create and save PCA visualization
    print("Creating PCA visualization...")
    pca_fig = create_pca_visualization(
        scaled_data, labels, numeric_cols, kmeans_model.cluster_centers_
    )
    pca_path = save_visualization(pca_fig, "pca_visualization.png")
    saved_files["pca"] = pca_path
    plt.close(pca_fig)

    # Create and save feature importance visualization
    print("Creating feature importance visualization...")
    feature_imp_fig = create_feature_importance_visualization(
        kmeans_model, numeric_cols, optimal_k
    )
    feature_imp_path = save_visualization(feature_imp_fig, "feature_importance.png")
    saved_files["feature_importance"] = feature_imp_path
    plt.close(feature_imp_fig)

    # Create and save parallel coordinates plot
    print("Creating parallel coordinates plot...")
    parallel_fig = create_parallel_coordinates(df_clean, numeric_cols, labels)
    parallel_path = save_visualization(parallel_fig, "parallel_coordinates.png")
    saved_files["parallel_coordinates"] = parallel_path
    plt.close(parallel_fig)

    # Create and save cluster distribution
    print("Creating cluster distribution visualization...")
    dist_fig, cluster_stats = create_cluster_distribution(df_clean, labels)
    dist_path = save_visualization(dist_fig, "cluster_distribution.png")
    saved_files["cluster_distribution"] = dist_path
    plt.close(dist_fig)

    # Create and save cluster profiles
    print("Creating cluster profiles visualization...")
    profiles_fig = create_cluster_profiles(df_clean, numeric_cols, labels, optimal_k)
    profiles_path = save_visualization(profiles_fig, "cluster_profiles.png")
    saved_files["cluster_profiles"] = profiles_path
    plt.close(profiles_fig)

    # Save the cluster assignments as CSV
    df_with_clusters = df_clean.copy()
    df_with_clusters["Cluster"] = labels

    # Create output directory if it doesn't exist
    output_dir = "kmeans_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Save cluster assignments
    cluster_csv_path = f"{output_dir}/cluster_assignments.csv"
    df_with_clusters.to_csv(cluster_csv_path, index=False)
    saved_files["cluster_assignments"] = cluster_csv_path

    # Save cluster statistics
    stats_csv_path = f"{output_dir}/cluster_statistics.csv"
    cluster_stats.to_csv(stats_csv_path)
    saved_files["cluster_statistics"] = stats_csv_path

    return saved_files


def main() -> None:
    """Main function to run the K-means visualization process."""
    file_path: str = "results.csv"

    try:
        # Load the results_summary.csv file
        df: pd.DataFrame = load_csv(file_path)
        print("results.py file loaded successfully!")

        # Display dataframe information
        display_dataframe_info(df)

        # Generate K-means visualizations
        print("\nGenerating K-means clustering visualizations...")
        saved_files = generate_kmeans_visualizations(df)

        if saved_files:
            print("\nVisualizations saved to:")
            for viz_type, path in saved_files.items():
                print(f"- {viz_type}: {path}")

            print("\nK-means clustering analysis complete!")
        else:
            print(
                "\nNo visualizations were generated. Please check the data requirements."
            )

    except Exception as e:
        print(f"Error during K-means clustering: {e}")
        print(f"Exception details: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        print("Please check the file path and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
