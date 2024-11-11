import logging
import joblib
from class_get_data import GetAndSplitData
from class_prerpocess_data import PreprocessData
from class_feature_processing import OneHotEncode, GenderToBinary
from class_train_model import Model
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    path = "sample_diabetes_mellitus_data.csv"
    data_loader = GetAndSplitData(path)

    # Load and split data
    df_train, df_test = data_loader.get_and_split_data()

    if df_train is None or df_test is None:
        logging.error("Data could not be loaded.")
        return

    # Data preprocessing
    preprocessor = PreprocessData(df_train)
    columns_to_fill = ['height', 'weight']
    columns_to_clean = ['age', 'gender', 'ethnicity']

    df_train = preprocessor.fill_with_mean(columns_to_fill)
    df_train = preprocessor.remove_na_rows(columns_to_clean)

    encoder = OneHotEncode()
    gender_transformer = GenderToBinary()

    df_train = encoder.transform(df_train, 'ethnicity')
    df_train = gender_transformer.transform(df_train)

    # Define features and target
    feature_columns = ['age', 'height', 'weight', 'aids', 'cirrhosis', 'hepatic_failure',
                       'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']
    target_column = 'diabetes_mellitus'

    # Instantiate the model with RandomForestClassifier instance
    model_instance_forest = Model(
        feature_columns=feature_columns,
        target_column=target_column,
        model=RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5),
        model_name="RandomForestClassifier"
    )

    # Train the model
    model_instance_forest.train(df_train)

    # Save the trained model
    model_filename = "random_forest_model.pkl"
    joblib.dump(model_instance_forest, model_filename)
    logging.info(f"Model saved as {model_filename}")
