# ===========================================================================================================================================
# 📊 MODEL EVALUATOR
# ===========================================================================================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    # Regression
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    median_absolute_error, explained_variance_score,
    # Classification
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold
from tabulate import tabulate

# ===========================================================================================================================================
# 📊 MAIN CLASS
# ===========================================================================================================================================
class ModelEvaluator:
    """
    A unified model evaluator for classification and regression tasks.

    Parameters
    ----------
    task_type : str
        "classification" or "regression".
    scoring_metric : str, optional
        Scoring metric used for plots/importance. Defaults to "f1" for classification, 
        "r2" for regression.
    generate_plots : bool, default=True
        Whether to generate diagnostic plots.
    minimum_precision : float, optional
        Minimum precision constraint for classification threshold optimization.
    minimum_recall : float, optional
        Minimum recall constraint for classification threshold optimization.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs.
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(
        self,
        task_type: str,
        scoring_metric: str = None,
        generate_plots: bool = True,
        minimum_precision: float = None,
        minimum_recall: float = None,
        random_state: int = 42,
        n_jobs: int = -1,
        output_dir: str = "output"
    ):
        self.task_type = task_type
        self.generate_plots = generate_plots
        self.minimum_precision = minimum_precision
        self.minimum_recall = minimum_recall
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.output_dir = output_dir

        # Default scoring
        if scoring_metric is None:
            if self.task_type == "classification":
                self.scoring_metric = "f1"
            elif self.task_type == "regression":
                self.scoring_metric = "r2"
            else:
                raise ValueError("task_type must be 'classification' or 'regression'")
        else:
            self.scoring_metric = scoring_metric

    ########################################################################################################################################
    ########################################################################################################################################
    # 📊 EVALUATE
    def evaluate(self, trained_models, X_train, X_test, y_train, y_test):
        """
        Evaluate trained models with metrics + plots.

        Parameters
        ----------
        trained_models : list
            Output from ModelTrainer.train()
            Each entry: (model_name, best_model, training_time, model_size_kb)

        Returns
        -------
        list
            Updated trained_models with appended metrics + evaluation time.
        """
        print(f"\n📊 Evaluating best {self.task_type} models...")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        results = []
        extra_data = {"roc": [], "pr": []} if self.task_type == "classification" else None
        feature_names = X_train.columns.tolist()

        for i, (model_name, best_model, training_time, model_size_kb) in enumerate(trained_models):
            print(f"\n   📋 Evaluating {model_name} model...")
            start_time = time.time()

            if self.task_type == "regression":
                metrics = self._compute_regression_metrics(model_name, best_model, X_train, y_train, X_test, y_test)
            else:
                metrics, roc_dict, pr_dict = self._compute_classification_metrics(
                    model_name, best_model, X_train, y_train, X_test, y_test
                )
                extra_data["roc"].append(roc_dict)
                extra_data["pr"].append(pr_dict)

            results.append(metrics)

            # 🔎 Feature importance
            self._generate_feature_importance(model_name, best_model, X_train, y_train, feature_names)

            # 📈 Plots
            if self.generate_plots:
                if self.task_type == "regression":
                    self._plot_error_diagnostics(model_name, best_model, X_test, y_test)
                else:
                    self._plot_confusion_matrix(model_name, best_model, X_test, y_test)
                    self._plot_calibration_curves(model_name, best_model, X_test, y_test)
                    self._generate_learning_curve(self.task_type, model_name, best_model, X_train, y_train)

            # ⏱️ Evaluation time
            evaluation_time = time.time() - start_time
            print(f"      └── Evaluation completed in {evaluation_time:.2f} seconds.")

            trained_models[i] = (model_name, best_model, training_time, model_size_kb, metrics, evaluation_time)

        # Save combined ROC/PR plots
        if self.task_type == "classification" and self.generate_plots:
            self._plot_combined_roc(extra_data["roc"])
            self._plot_combined_pr(extra_data["pr"])

        # Save evaluation metrics summary
        self._save_evaluation_metrics(results)

        return trained_models

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 REGRESSION HELPERS
    def _compute_regression_metrics(self, model_name, model, X_train, y_train, X_test, y_test):
        def _calc(X, y):
            y_pred = model.predict(X)
            y_true_log, y_pred_log = np.log1p(np.maximum(0, y)), np.log1p(np.maximum(0, y_pred))
            rmsle = np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
            return {
                "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
                "MAE": mean_absolute_error(y, y_pred),
                "MedianAE": median_absolute_error(y, y_pred),
                "R2": r2_score(y, y_pred),
                "Explained Variance": explained_variance_score(y, y_pred),
                "MAPE": mean_absolute_percentage_error(y, y_pred),
                "RMSLE": rmsle,
            }
        train, test = _calc(X_train, y_train), _calc(X_test, y_test)
        return {
            "Model": model_name,
            **{k: f"{test[k]:.3f} ({train[k]:.3f})" for k in test}
        }

    def _plot_error_diagnostics(self, model_name, model, X_test, y_test):
        try:
            print(f"      └── Plotting regression error diagnostics...")
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"{model_name} Diagnostics", fontsize=16)
            # Scatter Actual vs Pred
            axs[0,0].scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
            axs[0,0].plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
            axs[0,0].set_title("Predicted vs Actual")
            # Residuals vs Pred
            axs[0,1].scatter(y_pred, residuals, alpha=0.6, edgecolor="k")
            axs[0,1].axhline(0,color="red",ls="--")
            axs[0,1].set_title("Residuals vs Predicted")
            # Distribution
            sns.histplot(residuals, kde=True, ax=axs[1,0], bins=30, color="skyblue")
            axs[1,0].set_title("Error Distribution")
            # Residuals vs Actual
            axs[1,1].scatter(y_test, residuals, alpha=0.6, edgecolor="k")
            axs[1,1].axhline(0,color="red",ls="--")
            axs[1,1].set_title("Prediction Error Plot")
            plt.tight_layout(rect=[0,0.03,1,0.95])
            os.makedirs(f"{self.output_dir}/error_diagnostics", exist_ok=True)
            plt.savefig(f"{self.output_dir}/error_diagnostics/error_diagnostics_{model_name.replace(' ','_').lower()}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"      └── ⚠️ Failed diagnostics for {model_name}: {e}")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 CLASSIFICATION HELPERS
    def _compute_classification_metrics(self, model_name, model, X_train, y_train, X_test, y_test, default_threshold=0.5):
        print("      └── Computing classification performance metrics...")

        # Predict probs
        y_train_probs = model.predict_proba(X_train)[:,1]
        y_test_probs = model.predict_proba(X_test)[:,1]

        # Threshold optimization
        def find_threshold(y_true, y_probs):
            prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
            prec, rec = prec[:-1], rec[:-1]
            if self.minimum_precision and not self.minimum_recall:
                valid = prec >= self.minimum_precision
                if np.any(valid): return thresholds[np.argmax(rec*valid)]
            if self.minimum_recall and not self.minimum_precision:
                valid = rec >= self.minimum_recall
                if np.any(valid): return thresholds[np.argmax(prec*valid)]
            if self.minimum_precision and self.minimum_recall:
                valid = (prec>=self.minimum_precision)&(rec>=self.minimum_recall)
                if np.any(valid):
                    f1 = 2*(prec*rec)/(prec+rec+1e-8)
                    return thresholds[np.argmax(f1*valid)]
            return default_threshold

        threshold = find_threshold(y_train, y_train_probs)
        y_train_pred, y_test_pred = (y_train_probs>=threshold).astype(int), (y_test_probs>=threshold).astype(int)

        def scores(y_true,y_pred):
            return {"Accuracy": accuracy_score(y_true,y_pred),
                    "Precision": precision_score(y_true,y_pred),
                    "Recall": recall_score(y_true,y_pred),
                    "F1-Score": f1_score(y_true,y_pred)}

        train, test = scores(y_train,y_train_pred), scores(y_test,y_test_pred)

        # ROC & PR
        fpr,tpr,_ = roc_curve(y_test,y_test_probs); roc_auc=auc(fpr,tpr)
        prec,rec,_ = precision_recall_curve(y_test,y_test_probs); pr_auc=auc(rec,prec)

        return (
            {"Model": model_name, **{k:f"{test[k]:.3f} ({train[k]:.3f})" for k in test}, "ROC-AUC":f"{roc_auc:.3f}", "PR-AUC":f"{pr_auc:.3f}"},
            {"Model": model_name,"FPR":fpr,"TPR":tpr,"AUC":roc_auc},
            {"Model": model_name,"Precision":prec,"Recall":rec,"AUC-PR":pr_auc}
        )

    def _plot_confusion_matrix(self, model_name, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test,y_pred)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Negative","Positive"],yticklabels=["Negative","Positive"])
            plt.title(f"{model_name} Confusion Matrix")
            plt.savefig(f"{self.output_dir}/confusion_matrix_{model_name.replace(' ','_').lower()}.png",dpi=300)
            plt.close()
        except Exception as e:
            print(f"      └── ⚠️ Failed confusion matrix for {model_name}: {e}")

    def _plot_calibration_curves(self, model_name, model, X_test, y_test):
        try:
            prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:,1], n_bins=10)
            plt.figure(figsize=(6,5))
            plt.plot(prob_pred,prob_true,"s-")
            plt.plot([0,1],[0,1],"k--")
            plt.title(f"{model_name} Calibration Curve")
            plt.savefig(f"{self.output_dir}/calibration_{model_name.replace(' ','_').lower()}.png",dpi=300)
            plt.close()
        except Exception as e:
            print(f"      └── ⚠️ Failed calibration curve for {model_name}: {e}")

    def _plot_combined_roc(self, roc_data):
        plt.figure(figsize=(7,6))
        for d in roc_data:
            plt.plot(d["FPR"],d["TPR"],label=f"{d['Model']} (AUC={d['AUC']:.3f})")
        plt.plot([0,1],[0,1],"k--")
        plt.legend(); plt.title("ROC Curves")
        plt.savefig(f"{self.output_dir}/combined_roc.png",dpi=300); plt.close()

    def _plot_combined_pr(self, pr_data):
        plt.figure(figsize=(7,6))
        for d in pr_data:
            plt.plot(d["Recall"],d["Precision"],label=f"{d['Model']} (AUC={d['AUC-PR']:.3f})")
        plt.legend(); plt.title("Precision-Recall Curves")
        plt.savefig(f"{self.output_dir}/combined_pr.png",dpi=300); plt.close()

    ########################################################################################################################################
    ########################################################################################################################################
    # 📊 GENERATE FEATURE IMPORTANCE
    def _generate_feature_importance(self, model_name, best_model, X_train, y_train, feature_names):
        """
        Compute and save feature importance scores for the trained model.

        - Linear models → absolute coefficients.
        - Tree-based models → feature_importances_ attribute.
        - Other models → permutation importance (scoring = self.scoring_metric).

        Args:
            model_name (str): Name of the model.
            best_model (object): Trained sklearn model.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            feature_names (list): List of feature names.

        Returns:
            None. Saves feature importances to file.
        """
        print(f"      └── Generating feature importance scores for {model_name}...")

        try:
            # If model is a pipeline, get the actual estimator
            if hasattr(best_model, "named_steps"):
                final_estimator = best_model.named_steps.get("model", best_model)
            else:
                final_estimator = best_model

            # Coefficients (linear models)
            if hasattr(final_estimator, "coef_"):
                feature_importances = pd.Series(
                    abs(final_estimator.coef_[0]), index=feature_names
                ).sort_values(ascending=False)

            # Tree-based models
            elif hasattr(final_estimator, "feature_importances_"):
                feature_importances = pd.Series(
                    final_estimator.feature_importances_, index=feature_names
                ).sort_values(ascending=False)

            # Fallback → permutation importance
            else:
                perm_importance = permutation_importance(
                    final_estimator, X_train, y_train,
                    scoring=self.scoring_metric,
                    n_repeats=10,
                    random_state=self.random_state
                )
                feature_importances = pd.Series(
                    perm_importance.importances_mean, index=feature_names
                ).sort_values(ascending=False)
                
            # Save results
            os.makedirs(f"{self.output_dir}/feature_importance", exist_ok=True)
            file_path = f"{self.output_dir}/feature_importance/feature_importances_{model_name.replace(' ', '_').lower()}.txt"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"--- 📊 Feature Importance Scores for {model_name} ---\n")
                f.write(feature_importances.to_string())

            print(f"      └── Saved feature importance to {file_path}")

        except Exception as e:
            print(f"      └── ⚠️  Could not compute feature importance: {str(e)}")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📈 GENERATE LEARNING CURVES
    def _generate_learning_curve(self, task_type, model_name, model, X_train, y_train):
        """
        Generate and save a learning curve plot for the model.

        Args:
            task_type (str): "classification" or "regression".
            model_name (str): Model name (used in file naming).
            model (object): Trained sklearn model.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.

        Returns:
            None. Saves learning curve as PNG.
        """
        try:
            print(f"      └── Generating learning curve for {model_name}...")

            # CV Strategy
            if task_type == "classification":
                unique, counts = np.unique(y_train, return_counts=True)
                if np.min(counts) < 2:
                    print("      └── ⚠️ Not enough samples for some classes → using KFold instead.")
                    cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                else:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

            # Training sizes
            train_sizes = np.linspace(0.1, 1.0, 10)

            # Compute curves
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=cv,
                scoring=self.scoring_metric,
                n_jobs=self.n_jobs
            )

            train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
            test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color="b")
            plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.1, color="r")
            plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
            plt.plot(train_sizes, test_mean, 'o-', color="r", label="Cross-validation score")

            plt.title(f"Learning Curves - {model_name}", fontsize=14, fontweight="bold")
            plt.xlabel("Training Examples", fontsize=12)
            plt.ylabel(self.scoring_metric.upper(), fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(loc="best", fontsize=10)
            plt.tight_layout()

            # Save
            os.makedirs(f"{self.output_dir}/learning_curves", exist_ok=True)
            file_path = f"{self.output_dir}/learning_curves/learning_curve_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(file_path, dpi=300)
            plt.close()

            print(f"      └── Saved learning curve to {file_path}")

        except Exception as e:
            print(f"      └── ⚠️ Failed to generate learning curve: {e}")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📝 SAVE EVALUATION METRICS
    def _save_evaluation_metrics(self, results):
        """
        Save a consolidated evaluation metrics summary for all models.

        Args:
            results (list of dict): Each dict contains metrics for one model.

        Returns:
            None. Saves metrics to text file.
        """
        try:
            metrics_file_path = f"{self.output_dir}/evaluation_metrics_summary.txt"
            with open(metrics_file_path, "w", encoding="utf-8") as f:
                f.write("📋 Consolidated Evaluation Metrics:\n")
                f.write("(Test metrics first, training metrics in [brackets])\n\n")
                metrics_table = tabulate(
                    pd.DataFrame(results),
                    headers="keys",
                    tablefmt="grid",
                    floatfmt=".3f"
                )
                f.write(metrics_table + "\n\n")

            print(f"      └── Saved evaluation metrics summary to {metrics_file_path}")

        except Exception as e:
            print(f"      └── ⚠️ Failed to save evaluation metrics: {e}")