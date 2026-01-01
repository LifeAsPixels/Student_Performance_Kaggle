from Student_Performance_Kaggle import *

from pathlib import Path
import rich
from rich.console import Console
from rich.table import Table
from rich.traceback import install
install()

def main(): 

    # prepare configurations and arguments
    configuration = config.config()
    eda = EDA.EDA(configuration)
    viz_numerical = (eda.config.data_features_numerical,
                     eda.config.viz_path_parent,
                     )

    columns = ['study_hours', 'overall_score']
    viz_args = (columns, eda.config.viz_path_parent)
    # present some info about the classes being executed
    rich.print(eda.config.__doc__)
    rich.print(eda.__doc__)
    rich.print(f'Generating files...')
    
    # EDA: exploratory data analysis
    eda.data_overview(0, eda.config.report_path_parent)
    eda.boxplot_all(*viz_numerical)
    eda.boxplot(*viz_numerical)
    # rich.print(eda.config.report_path_parent, '\n\n', eda.config.viz_path_parent)
    eda.examine_outliers(eda.config.report_path_parent)
    eda.correlation_matrix_heatmap(eda.config.viz_path_parent)
    eda.scatter_plot(*viz_args)
    eda.elbow_method(*viz_args)

    # PreProcessor
    pp = preprocessor.preprocessor(configuration)
    pp.scale_features(columns)

    # Models
    model = models.ClusteringModel(k=3)
    labels = model.fit_predict(pp.scaled_data)

    model.final_clusters(configuration.df, columns, labels, configuration.viz_path_parent)

if __name__ == "__main__":
    main()