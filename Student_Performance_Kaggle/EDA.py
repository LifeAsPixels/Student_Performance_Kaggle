import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast
from Student_Performance_Kaggle import config
import inspect
import rich
from rich.progress import track
# from rich.console import Console

class EDA:
    '''
    Use the EDA class (Exploratory Data Analysis) to conduct simple data analysis
    on the data set to get acquainted with what the data looks like before
    preprocessing and cleaning.
    '''

    def __init__(self, cfg):
        self.cfg = config.config()
        self.df = self.cfg.df
        self.viz_preprocess()
        # Visualization cfgurations:
        self.color_scheme = 'skyblue'
        self.legend_bool = False
        
        
    def viz_preprocess(self):
        self.cfg.make_dir(self.cfg.viz_path_parent)
    
    def data_overview(self, count, save_path):
        with self.cfg.console.status('\nGenerating file(s)...\n', spinner='dots'):
            current_method = inspect.currentframe().f_code.co_name
            file = f'{save_path}\\{current_method}'
            file1 = f'{file}.csv'
            file2 = f'{file}.txt'
            file3 = f'{file}_head.csv'
            # get the metadata and save it
            info_df = pd.DataFrame({
                'Column': self.df.columns,
                'Non-Null Count': self.df.notnull().sum().values,
                'Null Count': self.df.isnull().sum().values,
                'Type': self.df.dtypes.values,
                })
            info_df.to_csv(file1, index=False)
            self.cfg.console.print(f'   [green]{file1}[/green]')

            # get the columns in list format and save it
            with open(file2, "w") as file:
                file.write(str(self.df.columns))
            self.cfg.console.print(f'   [green]{file2}[/green]')

            # get first few records as requested and save them
            if count > 0:
                self.df.head(count).to_csv(file3, index=False)
                self.cfg.console.print(f'   [green]{file3}[/green]')
            # return the values in case you want to print them in CLI
                return self.df.head(count), info_df, self.df.columns
            return info_df, self.df.columns

    def boxplot(self, columns, save_path, *args, **kwargs):
        """
        Creates a boxplot for each column in the provided list.
        """
        for col in track(columns, description='\nGenerating file(s)...\n'):
            current_method = inspect.currentframe().f_code.co_name
            file = f'{save_path}\\{current_method}_{col}.png'
            
            # 1. Initialize the figure (Canvas size)
            plt.figure(figsize=(10, 6))
            
            # 2. Create the boxplot
            # Using 'y' makes it vertical, which is better for spotting outliers
            sns.boxplot(data=self.df, y=col, color=self.color_scheme,  legend=self.legend_bool)
            
            # 3. Add titles and labels
            plt.title(f'Outlier Analysis: {col}')
            plt.ylabel('Value')
            
            # 4. Save the plot dynamically
            # Ensure the save_path exists before running
            plt.savefig(file)

            # 5. CRITICAL: Close the plot to free up RAM
            plt.close()
            rich.print(f'   [green]{file}[/green]')
    
    def boxplot_all(self, columns, save_path, *args, **kwargs):
        """
        Generates a single figure containing a grid of boxplots 
        for all numerical columns.
        """
        with self.cfg.console.status('\nGenerating file(s)...\n', spinner='dots'):
            current_method = inspect.currentframe().f_code.co_name
            file = f'{save_path}\\{current_method}.png'

            # 1. Determine the grid size (e.g., if you have 4 columns, 2x2)
            n = len(columns)
            ncols = 7
            nrows = (n // ncols) + (n % ncols)

            # 2. Create the Figure and the 'Axes' (the grid cells)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
            
            # Flatten axes into a 1D list so we can loop over them easily
            axes = axes.flatten()

            # 3. Loop through columns and assign each to a cell
            for i, col in enumerate(columns):
                sns.boxplot(data=self.df, y=col, ax=axes[i], color=self.color_scheme,  legend=self.legend_bool)
                axes[i].set_title(f'{col}')
            
            # 4. Remove any empty "cells" if the number of plots is odd
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # 5. Clean up the layout and save
            plt.tight_layout()
            plt.savefig(file)
            plt.close()
        self.cfg.console.print(f'   [green]{file}[/green]')
            

    def examine_outliers(self, save_path):
        with self.cfg.console.status('\nGenerating file(s)...\n', spinner='dots'):
            current_method = inspect.currentframe().f_code.co_name
            file = f'{save_path}\\{current_method}.csv'

            # 1. Get method name
            current_method = inspect.currentframe().f_code.co_name
            results = []

            # 2. Collect outliers
            for col in self.cfg.data_features_grades:
                results.append(self.df.query(f'{col} < 10'))

            # 3. Combine and Clean
            if results:
                # Concat and then remove duplicates based on the index
                zeros_df = pd.concat(results).drop_duplicates()
            else:
                zeros_df = pd.DataFrame(columns=self.df.columns)

            # 4. Secure Pathing (using an 'r' string or os.path.join)
            # index=False prevents the CSV from having an unnamed first column
            
            
        if not zeros_df.empty:
            zeros_df.to_csv(file, index=False)
            self.cfg.console.print(f'   [green]{file}[/green]')
        else:
            self.cfg.console.print(f'   [red]No outliers found.[/red] No file created.')

    def correlation_matrix_heatmap(self, save_path):
        '''
        Creates a heatmap on every possible pair of numerical columns
        using the Pearson Correlation Coefficient.
        '''
        with self.cfg.console.status('\nGenerating file(s)...\n', spinner='dots'):
            current_method = inspect.currentframe().f_code.co_name
            file = f'{save_path}\\{current_method}.png'

            # Only grab numeric columns
            numeric_df = self.df.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            
            # 2. Setup the figure
            plt.figure(figsize=(12, 10))
            
            # 3. Create the heatmap
            # 'annot=True' puts the numbers in the boxes
            # 'cmap' sets the colors (blue is negative, red is positive)
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            
            plt.title('Feature Correlation Matrix')
            plt.savefig(file)
            plt.close()
        self.cfg.console.print(f'   [green]{file}[/green]')

    def scatter_plot(self, x_col, y_col, save_path):
        '''
        Plot two numerical values against each other to visualize clusters or trends.
        '''
        with self.cfg.console.status('\nGenerating file(s)...\n', spinner='dots'):
            current_method = inspect.currentframe().f_code.co_name
            file = f'{save_path}\\{current_method}_{x_col}_vs_{y_col}.png'
            plt.figure(figsize=(10, 6))
            
            # 'alpha=0.5' makes the dots slightly see-through 
            # so you can see where they 'stack' on top of each other.
            sns.scatterplot(data=self.df, x=x_col, y=y_col, alpha=0.2, color='teal')
            
            plt.title(f'Relationship: {x_col} vs {y_col}')
            plt.grid(True, linestyle='--', alpha=0.5) # Adds a subtle grid
            
            plt.savefig(file)
            plt.close()
        self.cfg.console.print(f'   [green]{file}[/green]')
