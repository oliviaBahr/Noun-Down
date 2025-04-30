import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from typing import Optional, Dict
import numpy as np


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load results_summary.csv file using pandas.

    Args:
        file_path: Path to the results_summary.csv file

    Returns:
        Loaded pandas DataFrame
    """
    return pd.read_csv(file_path)


def display_dataframe_info(df: pd.DataFrame) -> None:
    """
    Display information about the DataFrame.

    Args:
        df: Pandas DataFrame to display information about
    """
    print("\nDataframe info:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head(5))


def get_top_entries(df: pd.DataFrame, column_name: str, n: int = 10) -> pd.DataFrame:
    """
    Get top n entries for the selected column based on values.

    Args:
        df: Input DataFrame
        column_name: Column to analyze
        n: Number of top entries to return

    Returns:
        DataFrame with top n entries
    """
    # Check if column is numeric
    if np.issubdtype(df[column_name].dtype, np.number):
        # For numeric columns, sort by value
        return df.sort_values(by=column_name, ascending=False).head(n)
    else:
        # For non-numeric columns, count occurrences and get top n
        value_counts = df[column_name].value_counts().reset_index()
        value_counts.columns = [column_name, "count"]
        return value_counts.head(n)


def create_visualization(
    top_data: pd.DataFrame, column_name: str, df: pd.DataFrame
) -> plt.Figure:
    """
    Create a visualization of the top entries with noun on y-axis
    and selected column values on x-axis.

    Args:
        top_data: DataFrame with the top entries
        column_name: Name of the column being visualized
        df: Original DataFrame (needed for some cases)

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(10, 8))  # Adjusted figure size for horizontal bars

    if "count" in top_data.columns:  # If we have a count column (from value_counts)
        # For categorical data, we need to get associated noun values
        if "noun" not in top_data.columns:
            # Join with original dataframe to get noun values
            merged_data = pd.DataFrame()
            for idx, row in top_data.iterrows():
                val = row[column_name]
                matching_rows = df[df[column_name] == val]
                top_nouns = matching_rows["noun"].value_counts().head(1)
                if not top_nouns.empty:
                    merged_data = pd.concat(
                        [
                            merged_data,
                            pd.DataFrame(
                                {
                                    "noun": top_nouns.index,
                                    column_name: val,
                                    "count": row["count"],
                                }
                            ),
                        ]
                    )

            if not merged_data.empty:
                top_data = merged_data.head(10)

        # Create horizontal bar chart (swapped axes)
        sns.barplot(y=top_data["noun"], x=top_data["count"], palette="viridis")
        plt.title(f"Top 10 values in '{column_name}' by count")
        plt.xlabel("Count")
    else:
        # Get the noun and selected column values
        sns.barplot(y=top_data["noun"], x=top_data[column_name], palette="viridis")
        plt.title(f"Top 10 entries in '{column_name}' (by {column_name} value)")
        plt.xlabel(column_name)

    plt.ylabel("Noun")
    plt.tight_layout()

    return fig


def create_statistics_table(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a table visualization showing statistics for all numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        Matplotlib figure object or None if no numeric columns
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if "noun" in numeric_cols:
        numeric_cols.remove("noun")  # Remove noun if it's somehow numeric

    if not numeric_cols:
        # If no numeric columns exist, create a simple info visual
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No numeric columns available for statistics visualization",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        fig.suptitle("DataFrame Statistics", fontsize=16)
        return fig

    # Calculate statistics for numeric columns
    stats_data = []
    for col in numeric_cols:
        stats = {
            "Column": col,
            "Mean": df[col].mean(),
            "Median": df[col].median(),
            "Std Dev": df[col].std(),
            "Min": df[col].min(),
            "Max": df[col].max(),
        }
        stats_data.append(stats)

    stats_df = pd.DataFrame(stats_data)

    # Create figure for statistics table
    fig, ax = plt.subplots(figsize=(10, len(numeric_cols) * 0.8 + 2))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=stats_df.iloc[:, 1:].round(2).values,
        rowLabels=stats_df["Column"].values,
        colLabels=stats_df.columns[1:],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("Statistical Summary of Numeric Columns", pad=20)
    plt.tight_layout()

    return fig


def create_individual_boxplots(df: pd.DataFrame) -> Dict[str, plt.Figure]:
    """
    Create individual boxplot visualizations for each numeric column.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of column names and their figure objects
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if "noun" in numeric_cols:
        numeric_cols.remove("noun")  # Remove noun if it's somehow numeric

    figures = {}

    # Create individual boxplot for each numeric column
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create boxplot with points for outliers
        sns.boxplot(y=df[col].dropna(), ax=ax, color="lightblue")

        # Add strip plot (individual points) over boxplot
        sns.stripplot(y=df[col].dropna(), ax=ax, color="darkblue", alpha=0.3, size=4)

        ax.set_title(f"Boxplot for {col}")
        ax.set_ylabel(col)

        figures[col] = fig

    return figures


def create_individual_distributions(df: pd.DataFrame) -> Dict[str, plt.Figure]:
    """
    Create individual distribution plots for each numeric column.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of column names and their figure objects
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if "noun" in numeric_cols:
        numeric_cols.remove("noun")  # Remove noun if it's somehow numeric

    figures = {}

    # Create individual distribution plot for each numeric column
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create histogram with KDE
        sns.histplot(
            df[col].dropna(),
            bins=min(20, len(df[col].unique())),
            kde=True,
            color="darkblue",
            alpha=0.6,
            ax=ax,
        )

        # Add median and mean lines
        median = df[col].median()
        mean = df[col].mean()

        ax.axvline(
            median,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median:.2f}",
        )
        ax.axvline(
            mean, color="green", linestyle="-", linewidth=1.5, label=f"Mean: {mean:.2f}"
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.legend()

        figures[col] = fig

    return figures


def create_correlation_matrix(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a correlation matrix visualization for numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        Matplotlib figure object or None if no numeric columns
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Need at least 2 numeric columns for correlation
    if len(numeric_cols) < 2:
        return None

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    plt.title("Correlation Matrix of Numeric Columns")
    plt.tight_layout()

    return fig


def save_visualization(fig: plt.Figure, filename: str) -> str:
    """
    Save the visualization as a PNG file.

    Args:
        fig: Matplotlib figure to save
        filename: Name of the file

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Generate full filepath
    filepath = f"{output_dir}/{filename}"

    # Save the figure
    fig.savefig(filepath, dpi=300, bbox_inches="tight")

    return filepath


def save_all_visualizations(df: pd.DataFrame, column_name: str) -> Dict[str, str]:
    """
    Generate and save all visualizations.

    Args:
        df: The DataFrame to visualize
        column_name: The specific column to visualize in detail

    Returns:
        Dictionary of visualization types and their file paths
    """
    saved_files = {}

    # 1. Create and save the column-specific visualization
    print("Generating column-specific visualization...")
    top_10 = get_top_entries(df, column_name, 10)
    column_fig = create_visualization(top_10, column_name, df)
    column_path = save_visualization(column_fig, f"top10_{column_name}_by_noun.png")
    saved_files["column_specific"] = column_path
    plt.close(column_fig)

    # 2. Create and save the statistics table
    print("Generating statistics table...")
    stats_fig = create_statistics_table(df)
    if stats_fig:
        stats_path = save_visualization(stats_fig, "statistical_summary.png")
        saved_files["statistics_table"] = stats_path
        plt.close(stats_fig)

    # 3. Create and save individual boxplots
    print("Generating individual boxplots...")
    boxplots = create_individual_boxplots(df)
    for col, fig in boxplots.items():
        safe_col_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
        boxplot_path = save_visualization(fig, f"boxplot_{safe_col_name}.png")
        saved_files[f"boxplot_{col}"] = boxplot_path
        plt.close(fig)

    # 4. Create and save individual distribution plots
    print("Generating individual distribution plots...")
    distributions = create_individual_distributions(df)
    for col, fig in distributions.items():
        safe_col_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
        dist_path = save_visualization(fig, f"distribution_{safe_col_name}.png")
        saved_files[f"distribution_{col}"] = dist_path
        plt.close(fig)

    # 5. Create and save the correlation matrix if possible
    print("Generating correlation matrix...")
    corr_fig = create_correlation_matrix(df)
    if corr_fig:
        corr_path = save_visualization(corr_fig, "correlation_matrix.png")
        saved_files["correlation"] = corr_path
        plt.close(corr_fig)

    return saved_files


def main() -> None:
    """Main function to run the visualization process."""
    file_path: str = "results.csv"

    try:
        # Load the input file
        df: pd.DataFrame = load_csv(file_path)
        print("results_summary.csv file loaded successfully!")

        # Display dataframe information
        display_dataframe_info(df)

        # Get the column name from command line arguments
        if len(sys.argv) > 1:
            column_to_visualize: str = sys.argv[1]
        else:
            # If no command line argument is provided, ask for user input
            print("\nPlease select a column to visualize from the list above.")
            column_to_visualize: str = input("Enter column name: ")

        # Check if the column exists
        if column_to_visualize in df.columns:
            print(f"Visualizing column: {column_to_visualize}")

            # Always generate and save the visualizations
            print("Generating visualizations...")
            saved_files = save_all_visualizations(df, column_to_visualize)

            print("\nVisualizations saved to:")
            for viz_type, path in saved_files.items():
                print(f"- {viz_type}: {path}")

            print("\nNote: Interactive display is not available.")
            print(
                "All visualizations have been saved as PNG files in the 'visualizations' directory."
            )

        else:
            print(f"Error: '{column_to_visualize}' is not a valid column name.")
            print(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)

    except Exception as e:
        print(f"Error loading or processing the results_summary.csv file: {e}")
        print(f"Exception details: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        print("Please check the file path and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
