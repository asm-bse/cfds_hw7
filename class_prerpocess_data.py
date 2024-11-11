
import logging
import pandas as pd

class PreprocessData:
    def __init__(self, df: pd.DataFrame):
        self.df = df


    def fill_with_mean(self, columns_to_fill):
        """
        Clean the data by removing unnecessary columns and rows.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with columns that need cleaning ('height', 'weight').
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with cleaned data.
        """     
        try:
            logging.info(f"Null values before filling with mean: {self.df.isnull().sum().sum()}")
            for column in columns_to_fill:
                if column in self.df.columns:
                    self.df.loc[:, column] = self.df[column].fillna(self.df[column].mean())
                else:
                    logging.warning(f"Column '{column}' not found in the DataFrame.")
            logging.info(f"Null values after filling with mean: {self.df.isnull().sum().sum()}")
        except KeyError:
            logging.error("The DataFrame does not contain the required columns.")
            return None
        return self.df
    
    
    def remove_na_rows(self, columns_to_clean):  
        """
        Clean the data by removing unnecessary columns and rows.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with columns that need cleaning ('age', 'gender', 'ethnicity').
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with cleaned data.
        """    
        try:
            logging.info(f"Rows before removing NaNs: {len(self.df)}")
            self.df = self.df.dropna(subset=columns_to_clean)
            logging.info(f"Rows after removing NaNs: {len(self.df)}")
        except KeyError:    
            logging.error("The DataFrame does not contain the required columns.")
            return None
        return self.df   

