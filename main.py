# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
from rich import print as rprint
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from config_models import models
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)

@ignore_warnings(category=Warning)
def main():

    # Configuration file path
    config_path = "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load CSV file into a DataFrame
    rprint("\n[bold blue]📥 Loading dataset for processing...[/bold blue]")
    data_loader = DataLoader()
    df = data_loader.load_csv(filepath=config["data_file_path"])

    # Initialize and run data cleaning
    rprint("\n[bold blue]🧼 Cleaning dataset...[/bold blue]")
    data_cleaner = DataCleaner(config=config)
    df_cleaned = data_cleaner.clean_all(df=df)

    # Initialize and run data preprocessing
    rprint("\n[bold blue]⚙️  Preprocessing dataset...[/bold blue]")
    data_preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = data_preprocessor.split_dataset(df=df_cleaned, target=config["target_column"], stratify=False)
    X_train = data_preprocessor.fit_transform(df=X_train)
    X_test = data_preprocessor.transform(df=X_test)

    # Train the models
    rprint("\n[bold blue]🤖 Training candidate models...[/bold blue]")
    model_trainer = ModelTrainer(
        task_type="regression",   # or "regression"
        search_type="grid",         # "grid", "random", or "bayes"
        use_smote_enn=False,          # disable for iris (balanced dataset)
        cv_folds=3,
        n_iter=10,                    # only for random/bayes
        scoring_metric=None,          
        random_state=42
    )
    trained_models = model_trainer.train(models, X_train, y_train)

    # Evaluate the models
    model_evaluator = ModelEvaluator(
        task_type="regression",   # or "regression"
        generate_plots=True,
        scoring_metric=None,          
        random_state=42,
        n_jobs=1
    )   
    model_evaluator.evaluate(
        trained_models=trained_models, 
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test
    )

if __name__ == "__main__":
    main()
