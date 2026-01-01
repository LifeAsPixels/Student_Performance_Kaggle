import kagglehub
import os
import sys
import shutil
from pathlib import Path
import inspect
import pandas as pd
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import track

class config:
    """
    Use the config class to download the data and handle its IO.
    """
    def __init__(self):
        # initialize the Rich console for fancy CLI outputs
        self.console = Console()

        self.df = None
        # define all the paths and filenames used for the data and reports
        self.data_path_kaggle = 'kundanbedmutha/student-performance-dataset'
        self.data_path_original = Path(r'~\.cache\kagglehub\datasets\kundanbedmutha\student-performance-dataset\versions\1\Student_Performance.csv').expanduser()
        self.data_path_waste = Path(r'~\.cache\kagglehub').expanduser()
        self.data_path_parent = Path(os.path.join(self.caller_script_dir(), Path('data\\')))
        self.data_path_name = "Student_Performance.csv"
        self.data_path_absolute = Path(os.path.join(self.data_path_parent, self.data_path_name))
        self.data_path_url = 'https://www.kaggle.com/datasets/kundanbedmutha/student-performance-dataset'
        self.report_path_parent = Path(os.path.join(self.caller_script_dir(), Path('report\\')))
        self.viz_path_parent = Path(os.path.join(self.report_path_parent, 'viz\\'))

        # download the data and load it to memory        
        self.download_data_relative()
        self.df = pd.read_csv(self.data_path_absolute)
        
        # define columns sets based on their qualities
        self.feature__label_definitions()

    def download_unnecessary(self):
        return self.data_path_absolute.exists()

    def delete_source_folder(self):
        if self.data_path_waste.exists():
            print('\nRemoving:\n  ', self.data_path_waste)
            shutil.rmtree(self.data_path_waste)
    
    def caller_script_dir(self):
        # walk the stack and find the first frame whose filename is not this file
        this_file = Path(__file__).resolve()
        for frame_info in inspect.stack()[1:]:
            caller_path = Path(frame_info.filename).resolve()
            if caller_path != this_file:
                return caller_path.parent
        return None

    def make_dir(self, path):
        path.mkdir(parents=True, exist_ok=True)
        return None
    
    def download_data_relative(self):
        if self.download_unnecessary():
            print()
            return None
        # delete file cache if it already exists
        self.delete_source_folder()
        # Download latest version
        with self.console.status("Downloading...", spinner="dots"):
            kagglehub.dataset_download(self.data_path_kaggle)
        # move file to relative script path and delete the original DL folder
        src = self.data_path_original
        dst = self.data_path_parent
        print("\nData downloaded to:\n  ", dst)
        self.make_dir(dst)
        shutil.copy2(src, dst)
        self.delete_source_folder()
        # load the csv to pandas dataframe
        print('\nPath to data file(s):\n  ', self.data_path_absolute)
    
    def feature__label_definitions(self):
        self.data_features_categorical = ['gender',
                                'school_type',
                                'parent_education',
                                'internet_access',
                                'travel_time',
                                'extra_activities',
                                'study_method',
                                ]
        self.data_features_numerical = ['student_id',
                              'age',
                              'study_hours',
                              'attendance_percentage',
                              'math_score',
                              'science_score',
                              'english_score',
                              ]
        self.data_features_grades = ['math_score',
                              'science_score',
                              'english_score',
                              ]
        self.data_labels = ['overall_score', 'final_grade']

