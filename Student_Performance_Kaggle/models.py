from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import rich
from rich.progress import track
import inspect

class ClusteringModel:
    def __init__(self, k=3):
        self.k = k
        self.model = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        self.clusters = None

    def fit_predict(self, scaled_df):
        """
        Runs the K-Means algorithm and returns the cluster labels.
        """
        self.clusters = self.model.fit_predict(scaled_df)
        return self.clusters

    def final_clusters(self, df, list_xcol_ycol, cluster_labels, save_path, *args):
        """
        Creates a scatter plot colored by the identified clusters.
        """
        x_col, y_col = list_xcol_ycol
        current_method = inspect.currentframe().f_code.co_name
        file = f'{save_path}\\{current_method}_{x_col}_vs_{y_col}.png'
        plt.figure(figsize=(10, 7))
        
        # We plot the ORIGINAL values for x and y so the axis numbers make sense,
        # but we use the CLUSTER LABELS for the color (hue).
        sns.scatterplot(
            data=df, 
            x=x_col, 
            y=y_col, 
            hue=cluster_labels, 
            palette='viridis', 
            s=100, 
            alpha=0.7
        )
        
        plt.title(f'Student Segments based on {x_col} and {y_col}')
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.legend(title='Cluster Group')
        
        plt.savefig(file)
        plt.close()
        rich.print(f'   [green]{file}[/green]')