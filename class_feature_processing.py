import logging
import pandas as pd
from abc import ABCMeta, abstractmethod


class FeatureTransformer(metaclass=ABCMeta):
    """
    Abstract base class for feature transformation.
    All feature transformers must inherit from this class and implement the transform method.
    """
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature transformation to the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame to transform.
        
        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame.
        """
        pass


class OneHotEncode(FeatureTransformer):
    """
    A class to one-hot encode the given column.
    """
    def transform(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Apply one-hot encoding to the given column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with one-hot encoded given column.
        """
        try:
            logging.info(f"Starting one-hot encoding for '{column_name}' column.")
            
            if column_name in df.columns:
                df = pd.get_dummies(df, columns=[column_name], drop_first=True)
                logging.info(f"'{column_name}' column successfully one-hot encoded.")
            else:
                logging.warning(f"'{column_name}' column not found in the DataFrame.")
            
        except Exception as e:
            logging.error(f"An error occurred during one-hot encoding: {e}")
        
        return df


class GenderToBinary(FeatureTransformer):
    """
    A class to map the 'gender' column to binary values ('M' -> 1, 'F' -> 0).
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the 'gender' column to binary values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with the 'gender_binary' column.
        """
        try:
            logging.info("Starting transformation of 'gender' column to binary.")
            
            if 'gender' in df.columns:
                df.loc[:, 'gender_binary'] = df.loc[:, 'gender'].map({'M': 1, 'F': 0})
                df.drop('gender', axis=1, inplace=True)
                logging.info("'gender' column successfully mapped to binary values.")
            else:
                logging.warning("'gender' column not found in the DataFrame.")
            
        except Exception as e:
            logging.error(f"An error occurred during gender-to-binary transformation: {e}")
        
        return df
