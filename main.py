# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
from rich import print as rprint
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_explorer import DataExplorer
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.data_preprocessor import DataPreprocessor
from src.model_candidates import models
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
@ignore_warnings(category=Warning)
def main():

    # Configuration file path
    config_path = "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    ###############################################################################################################################
    ###############################################################################################################################
    # 📥 DATA LOADING
    # - Load CSV file into a DataFrame
    rprint("\n[bold blue]📥 Loading dataset for processing...[/bold blue]")
    data_loader = DataLoader()
    df = data_loader.load_csv(filepath=config["data_file_path"])

    ###############################################################################################################################
    ###############################################################################################################################
    # 🧼 DATA CLEANING
    # - Initialize and run data cleaning
    rprint("\n[bold blue]🧼 Cleaning dataset...[/bold blue]")
    data_cleaner = DataCleaner(config=config)
    df_cleaned = data_cleaner.clean_all(df=df)

    ###############################################################################################################################
    ###############################################################################################################################
    # 🔎 DATA CHANGES (RAW VS CLEANED)
    # - Compare raw vs cleaned data
    data_explorer = DataExplorer()
    data_explorer.compare_dataframes(df_before=df, df_after=df_cleaned, before_name="Raw Data", after_name="Cleaned Data")

    ###############################################################################################################################
    ###############################################################################################################################
    # ⚙️ DATA PREPROCESSING
    # - Initialize and run data preprocessing
    rprint("\n[bold blue]⚙️  Preprocessing dataset...[/bold blue]")
    data_preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = data_preprocessor.split_dataset(df=df_cleaned, target=config["target_column"], stratify=False)
    X_train = data_preprocessor.fit_transform(df=X_train)
    X_test = data_preprocessor.transform(df=X_test)

    ###############################################################################################################################
    ###############################################################################################################################
    # 🔎 DATA CHANGES (CLEANED VS PREPROCESSED) 
    # - Compare cleaned vs preprocessed data
    df_train_transformed = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    df_test_transformed = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    df_transformed = pd.concat([df_train_transformed, df_test_transformed], axis=0).reset_index(drop=True)  
    data_explorer.compare_dataframes(df_before=df_cleaned, df_after=df_transformed, before_name="Cleaned Data", after_name="Preprocessed Data")

    ###############################################################################################################################
    ###############################################################################################################################
    # 🤖 MODEL TRAINING
    # - Train the models
    rprint(f"\n[bold blue]🤖 Training candidate {config["task_type"]} models...[/bold blue]")
    model_trainer = ModelTrainer(config=config)
    trained_models = model_trainer.train(models=models, X_train=X_train, y_train=y_train)

    ###############################################################################################################################
    ###############################################################################################################################
    # 📊 MODEL EVALUATION
    # - Evaluate the models
    rprint(f"\n[bold blue]📊 Evaluating best {config["task_type"]} models...[/bold blue]")
    model_evaluator = ModelEvaluator(config=config)   
    model_evaluator.evaluate(
        trained_models=trained_models, 
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test
    )

if __name__ == "__main__":
    main()
