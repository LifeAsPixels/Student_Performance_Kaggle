from Student_Performance_Kaggle import *
import rich
from rich.console import Console
from rich.table import Table
from rich.traceback import install
install()

def main(): 
    configuration = config.config()
    eda = EDA.EDA(configuration)
    viz_numerical = (
        eda.cfg.data_features_numerical,
        eda.cfg.viz_path_parent
        )

    exploratory_data_analysis = [
        lambda: rich.print(eda.cfg.__doc__),
        lambda: rich.print(eda.__doc__),
        lambda: eda.data_overview(0, eda.cfg.report_path_parent),
        lambda: eda.boxplot_all(*viz_numerical),
        lambda: eda.boxplot(*viz_numerical),
        # lambda: rich.print(eda.cfg.report_path_parent, '\n\n', eda.cfg.viz_path_parent),
        lambda: eda.examine_outliers(eda.cfg.report_path_parent),
        lambda: eda.correlation_matrix_heatmap(eda.cfg.viz_path_parent),
    ]

    groups_to_run = [
        exploratory_data_analysis,
    ]

    for group in groups_to_run:
        if callable(group):
            group()
        else:
            for action in group:
                action()

if __name__ == "__main__":
    main()