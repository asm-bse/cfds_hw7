import logging 
import pandas as pd
from sklearn.model_selection import train_test_split


class GetAndSplitData:
    def __init__(self, path, test_size=0.2, random_state=42):
        self.path = path
        self.test_size = test_size
        self.random_state = random_state


    def get_and_split_data(self):
        """
        Load a CSV file into a pandas DataFrame with error handling and logging.
        
        Parameters:
        -----------
        path : str
            The path to the CSV file to be loaded.
        
        Returns:
        --------
        df : pd.DataFrame or None
            Returns the loaded DataFrame if successful, None otherwise.
        """
        try:
            # Logging the start of data loading
            logging.info(f"Loading data from file: {self.path}")
            
            # Load the data
            df = pd.read_csv(self.path)

            # Logging the success of data loading
            logging.info(f"Data successfully loaded. Number of rows: {len(df)}, number of columns: {len(df.columns)}")
            
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
            
            # Logging the success of data splitting
            logging.info(f"Data successfully split into train and test sets. "
                         f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

            return train_df, test_df


        except FileNotFoundError:
            # Logging an error if the file is not found
            logging.error(f"File not found: {self.path}")
            return None, None

        except pd.errors.EmptyDataError:
            # Logging an error if the file is empty
            logging.error(f"File is empty: {self.path}")
            return None, None

        except pd.errors.ParserError:
            # Logging an error if there are issues with parsing the CSV
            logging.error(f"Error parsing the file: {self.path}")
            return None, None

        except Exception as e:
            # Logging any other error
            logging.error(f"An error occurred while loading data from the file: {self.path}. Error: {e}")
            return None, None