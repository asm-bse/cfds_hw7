from sklearn.metrics import roc_auc_score
import numpy as np
import logging

class Model:
    def __init__(self, feature_columns, target_column, model, model_name="Model", **model_params):
        self._feature_columns = feature_columns
        self._target_column = target_column

        # Initialize the model instance
        try:
            self.model = model  # Ensure model is an instance, not a class
            logging.info(f"Model {model_name} initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing model {model_name}: {e}")
            raise

        self.model_name = model_name

    def train(self, df):
        try:
            X = df[self._feature_columns]
            y = df[self._target_column]

            self.model.fit(X, y)
            logging.info(f"Model {self.model_name} successfully trained.")
        
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise
    
    def predict(self, df):
        try:
            X = df[self._feature_columns]
            predictions = self.model.predict(X)
            logging.info(f"Prediction with model {self.model_name} successful.")
            return predictions

        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise

    def get_roc_auc(self, y_train, y_test, train_predictions, test_predictions, round_digits=2, verbose=False):
        try:
            roc_auc_train = roc_auc_score(y_train, train_predictions).round(round_digits)
            roc_auc_test = roc_auc_score(y_test, test_predictions).round(round_digits)

            if verbose:
                logging.info(f"ROC AUC for Training Set: {roc_auc_train}")
                logging.info(f"ROC AUC for Test Set: {roc_auc_test}")
            
            return roc_auc_train, roc_auc_test
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            return None, None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None, None
