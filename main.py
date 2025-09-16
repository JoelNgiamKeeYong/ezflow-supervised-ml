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
from src.model_candidates import models, models_debug_classification, models_debug_regression
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
@ignore_warnings(category=Warning)

def main():

    # Configuration file path
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    ###############################################################################################################################
    ###############################################################################################################################
    # 📥 DATA LOADING
    rprint("\n[bold blue]📥 Loading dataset for processing[/bold blue]...")
    data_loader = DataLoader()  
    if not config["debug"]: 
        # Load project dataset
        df = data_loader.load_csv(filepath=config["data_file_path"])  
    else:
        # Load sample dataset
        df = data_loader.load_sklearn_sample(task_type=config["debug_task_type"]) 

    ###############################################################################################################################
    ###############################################################################################################################
    # 🧼 DATA CLEANING
    rprint("\n[bold blue]🧼 Cleaning dataset[/bold blue]...")
    data_cleaner = DataCleaner()  
    if not config["debug"]:
        # Apply project specific cleaning steps
        df_cleaned = data_cleaner.clean_all(df=df)  
    else:
        # Skip cleaning process; assume dataset is ready for preprocessing
        print(f"   └── Skipping cleaning process...")  
        df_cleaned = df.copy()
        
    ###############################################################################################################################
    ###############################################################################################################################
    # 🔎 DATA CHANGES (RAW VS CLEANED)
    data_explorer = DataExplorer()  
    data_explorer.compare_dataframes(df_before=df, df_after=df_cleaned, before_name="Raw Data", after_name="Cleaned Data")

    ###############################################################################################################################
    ###############################################################################################################################
    # ⚙️ DATA PREPROCESSING
    rprint("\n[bold blue]⚙️  Preprocessing dataset[/bold blue]...")
    data_preprocessor = DataPreprocessor()  

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = data_preprocessor.split_dataset(
        df=df_cleaned,
        target=config["target_column"] if not config["debug"] else "target",
        test_size=config["test_size"],
        stratify=config["stratify"] if not config["debug"] else (True if config["debug_type"] == "classification" else False),
        random_state=config["random_state"],
    )
    
    # Transform data
    X_train = data_preprocessor.fit_transform(df=X_train)
    X_test = data_preprocessor.transform(df=X_test)

    ###############################################################################################################################
    ###############################################################################################################################
    # 🔎 DATA CHANGES (CLEANED VS PREPROCESSED)
    df_train_transformed = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    df_test_transformed = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    df_transformed = pd.concat([df_train_transformed, df_test_transformed], axis=0).reset_index(drop=True)  
    data_explorer.compare_dataframes(df_before=df_cleaned, df_after=df_transformed, before_name="Cleaned Data", after_name="Preprocessed Data")

    ###############################################################################################################################
    ###############################################################################################################################
    # 🤖 MODEL TRAINING
    rprint(f"\n[bold blue]🤖 Training candidate {config['task_type']} models[/bold blue]...")
    model_trainer = ModelTrainer(  
        task_type=config["task_type"] if not config["debug"] else config["debug_task_type"],
        resampling_method=config["resampling_method"],
        hyperparameter_search_type=config["hyperparameter_search_type"],
        cv_folds=config["cv_folds"],
        scoring_metric=config["scoring_metric"] if not config["debug"] else ("accuracy" if config["debug_task_type"] == "classification" else "r2"),
        n_iter=config["n_iter"],
        n_jobs=config["n_jobs"],
        random_state=config["random_state"],
        model_dir=config["model_dir"]
    )
    trained_models = model_trainer.train(
        models=models if not config["debug"] \
            else (models_debug_classification if config["debug_task_type"] == "classification" \
                  else models_debug_regression),
        X_train=X_train, y_train=y_train
    )

    ###############################################################################################################################
    ###############################################################################################################################
    # 📊 MODEL EVALUATION
    rprint(f"\n[bold blue]📊 Evaluating best {config['task_type']} models[/bold blue]...")
    model_evaluator = ModelEvaluator(
        task_type=config["task_type"] if not config["debug"] else config["debug_task_type"],
        scoring_metric=config["scoring_metric"] if not config["debug"] else ("accuracy" if config["debug_task_type"] == "classification" else "r2"),
        minimum_precision=config["minimum_precision"],
        minimum_recall=config["minimum_recall"],
        generate_plots=config["generate_plots"],
        n_jobs=config["n_jobs"],
        random_state=config["random_state"],
        output_dir=config["output_dir"]
    )  
    model_evaluator.evaluate(trained_models=trained_models, X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    main()
