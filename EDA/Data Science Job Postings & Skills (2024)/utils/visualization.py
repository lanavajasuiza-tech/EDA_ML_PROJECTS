"""
Data visualization functions for exploratory analysis and clustering results.

This module contains tools to create charts that facilitate
the interpretation of clustering and data analysis results. It includes:
- Scatter plots for clusters.
- Heatmaps for feature correlations.
- Boxplots for data distributions across clusters.
- Bar plots for averages or frequencies per cluster.
- Combined visualization for an overview.

Classes:
1. HeatmapVisualizer: Creates heatmaps for correlation analysis.
2. ScatterplotVisualizer: Plots scatter diagrams for clusters.
3. BarplotVisualizer: Visualizes means or frequencies using bar plots.
4. BoxplotVisualizer: Shows data distributions with boxplots.
5. DataVisualizationCoordinator: Combines all visualizations into one layout.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# =================== HEATMAP VISUALIZER =================== #

class HeatmapVisualizer:
    def __init__(self, df):
        self.df = df

    def plot(self, title="Correlation Heatmap"):
        """
        Generates a heatmap of feature correlations.

        Parameters:
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.title(title)
        plt.show()


# =================== SCATTERPLOT VISUALIZER =================== #

class ScatterplotVisualizer:
    def __init__(self, df):
        self.df = df

    def plot(self, x, y, clusters, title="Scatter Plot"):
        """
        Creates a scatter plot with clusters.

        Parameters:
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            clusters (str): Column name for cluster labels.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[x], y=self.df[y], hue=self.df[clusters], palette='viridis', s=100, alpha=0.7)
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(title='Cluster')
        plt.show()


# =================== BARPLOT VISUALIZER =================== #

class BarplotVisualizer:
    def __init__(self, df):
        self.df = df

    def plot(self, column, clusters, title="Bar Plot"):
        """
        Creates a bar plot showing means by cluster.

        Parameters:
            column (str): Column name to analyze.
            clusters (str): Column name for cluster labels.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        self.df.groupby(clusters)[column].mean().plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.xlabel('Cluster')
        plt.ylabel(f'Mean of {column}')
        plt.show()


# =================== BOXPLOT VISUALIZER =================== #

class BoxplotVisualizer:
    def __init__(self, df):
        self.df = df

    def plot(self, column, clusters, title="Boxplot by Cluster"):
        """
        Creates a boxplot to show distributions by cluster.

        Parameters:
            column (str): Column name to analyze.
            clusters (str): Column name for cluster labels.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=clusters, y=column, data=self.df, palette='viridis')
        plt.title(title)
        plt.xlabel('Cluster')
        plt.ylabel(column)
        plt.show()


# =================== DATA VISUALIZATION COORDINATOR =================== #

class DataVisualizationCoordinator:
    def __init__(self, df):
        self.df = df
        self.heatmap = HeatmapVisualizer(df)
        self.scatterplot = ScatterplotVisualizer(df)
        self.barplot = BarplotVisualizer(df)
        self.boxplot = BoxplotVisualizer(df)

    def plot_all(self, x, y, column, clusters):
        """
        Creates a combined visualization with all charts.

        Parameters:
            x (str): Column name for x-axis in scatter plot.
            y (str): Column name for y-axis in scatter plot.
            column (str): Column name for bar plot and boxplot.
            clusters (str): Column name for cluster labels.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Comprehensive Visualizations", fontsize=16)

        # Heatmap
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', ax=axes[0, 0])
        axes[0, 0].set_title("Correlation Heatmap")

        # Scatter plot
        sns.scatterplot(x=self.df[x], y=self.df[y], hue=self.df[clusters], palette='viridis', ax=axes[0, 1], s=100, alpha=0.7)
        axes[0, 1].set_title("Scatter Plot")

        # Bar plot
        self.df.groupby(clusters)[column].mean().plot(kind='bar', ax=axes[1, 0], color='skyblue')
        axes[1, 0].set_title("Bar Plot")
        axes[1, 0].set_xlabel("Cluster")
        axes[1, 0].set_ylabel(f"Mean of {column}")

        # Boxplot
        sns.boxplot(x=clusters, y=column, data=self.df, palette='viridis', ax=axes[1, 1])
        axes[1, 1].set_title("Boxplot by Cluster")
        axes[1, 1].set_xlabel("Cluster")
        axes[1, 1].set_ylabel(column)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()