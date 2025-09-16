# ===========================================================================================================================================
# 📊 MODEL EVALUATOR
# ===========================================================================================================================================

import os
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
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
from rich import print as rprint

# ===========================================================================================================================================
# 📊 MAIN CLASS
# ===========================================================================================================================================
class ModelEvaluator:
    """
    A unified model evaluator for classification and regression tasks.

    Parameters
    ----------
    task_type : str
        Type of task: "classification" or "regression".
    scoring_metric : str
        Metric to optimize (e.g., "accuracy", "r2").
    output_dir : str
        Directory to save evaluation outputs.
    generate_plots : bool, default=True
        Whether to generate and save evaluation plots.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs.
    minimum_precision : float, optional
        Precision constraint for threshold tuning (classification).
    minimum_recall : float, optional
        Recall constraint for threshold tuning (classification).
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(
        self,
        task_type: str = "classification",
        scoring_metric: str = "accuracy",
        output_dir: str = "output",
        generate_plots: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        minimum_precision: Optional[float] = None,
        minimum_recall: Optional[float] = None,
    ):
        self.task_type = task_type
        self.scoring_metric = scoring_metric
        self.output_dir = output_dir
        self.generate_plots = generate_plots
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.minimum_precision = minimum_precision
        self.minimum_recall = minimum_recall

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
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        results = []
        extra_data = {"roc": [], "pr": []} if self.task_type == "classification" else None
        feature_names = X_train.columns.tolist()

        # Evaluate each model
        for i, (model_name, best_model, training_time, model_size_kb) in enumerate(trained_models):
            print(f"\n   📋 Evaluating \033[1;38;5;214m{model_name}\033[0m model...")
            start_time = time.time()

            # Compute metrics
            if self.task_type == "regression":
                # Compute regression metrics
                metrics = self._compute_regression_metrics(model_name, best_model, X_train, y_train, X_test, y_test)

            else:
                # Compute classification metrics
                metrics, roc_dict, pr_dict = self._compute_classification_metrics(model_name, best_model, X_train, y_train, X_test, y_test)

                # Appending ROC and PR data
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
            self._plot_roc_curves(extra_data["roc"])
            self._plot_pr_curves(extra_data["pr"])

        # Save evaluation metrics summary
        self._save_evaluation_metrics(results)

        return trained_models

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 COMPUTE REGRESSION METRICS
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

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 PLOT ERROR DIAGNOSTICS
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

            # Save results
            os.makedirs(f"{self.output_dir}/error_diagnostics", exist_ok=True)
            file_path = f"{self.output_dir}/error_diagnostics/error_diagnostic{model_name.replace(' ', '_').lower()}.png"
            plt.tight_layout(rect=[0,0.03,1,0.95])
            plt.savefig(file_path, dpi=300)
            plt.close()

            rprint(f"      └── Saved error diagnostics to '{file_path}'")
            
        except Exception as e:
            print(f"      └── ⚠️  Failed diagnostics for {model_name}: {e}")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 COMPUTE CLASSIFICATION METRICS
    def _compute_classification_metrics(self,model_name, model,X_train, y_train, X_test, y_test,default_threshold=0.5):
        """
        Computes classification metrics, ROC, and Precision-Recall (PR) curves for a given model.

        Parameters
        ----------
        model_name : str
            Name of the model for logging and plotting.
        model : object
            Trained scikit-learn compatible model with `predict` and optionally `predict_proba`.
        X_train : array-like of shape (n_samples, n_features)
            Training input features.
        y_train : array-like of shape (n_samples,)
            Training labels.
        X_test : array-like of shape (n_samples, n_features)
            Test input features.
        y_test : array-like of shape (n_samples,)
            Test labels.
        default_threshold : float, default=0.5
            Default threshold for binary classification if no custom threshold is selected.

        Returns
        -------
        metrics : dict
            Dictionary with overall metrics including Accuracy, Precision, Recall, F1-Score
            in the format "test (train)" for easy reporting.
        roc_dict : dict or None
            ROC curve data including FPR, TPR, and AUC. 
            For multiclass, returned as a dict of classes.
        pr_dict : dict or None
            Precision-Recall curve data including Precision, Recall, and AUC-PR.
            For multiclass, returned as a dict of classes.

        Notes
        -----
        - Supports threshold optimization for binary classifiers based on minimum precision/recall.
        - Uses one-vs-rest strategy for multiclass ROC and PR curves.
        """
        print("      └── Computing classification performance metrics...")

        # Determine classes
        classes = getattr(model, "classes_", np.unique(y_train))
        is_binary = len(classes) == 2

        # Initialize outputs
        roc_dict = None
        pr_dict = None

        # ---------------- Binary Classification ----------------
        if is_binary and hasattr(model, "predict_proba"):
            y_train_probs = model.predict_proba(X_train)[:, 1]
            y_test_probs  = model.predict_proba(X_test)[:, 1]

            # Threshold optimization (optional)
            def find_threshold(y_true, y_probs):
                prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
                prec, rec = prec[:-1], rec[:-1]
                # Simple F1-maximizing threshold
                f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
                if np.any(f1_scores):
                    return thresholds[np.argmax(f1_scores)]
                return default_threshold

            threshold = find_threshold(y_train, y_train_probs)
            y_train_pred = (y_train_probs >= threshold).astype(int)
            y_test_pred  = (y_test_probs >= threshold).astype(int)

            # ROC / PR metrics
            fpr, tpr, _ = roc_curve(y_test, y_test_probs)
            prec, rec, _ = precision_recall_curve(y_test, y_test_probs)
            roc_dict = {"Model": model_name, "FPR": fpr, "TPR": tpr, "AUC": auc(fpr, tpr)}
            pr_dict  = {"Model": model_name, "Precision": prec, "Recall": rec, "AUC-PR": auc(rec, prec)}

            roc_macro = roc_micro = roc_dict["AUC"]
            pr_macro  = pr_micro  = pr_dict["AUC-PR"]

        # ---------------- Multiclass Classification ----------------
        else:
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)
            y_train_probs = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
            y_test_probs  = model.predict_proba(X_test)  if hasattr(model, "predict_proba") else None

            roc_dict, pr_dict = {}, {}
            fpr_all, tpr_all = [], []
            prec_all, rec_all = [], []

            for i, cls in enumerate(classes):
                y_test_bin = (y_test == cls).astype(int)
                y_score = y_test_probs[:, i]
                fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                prec, rec, _ = precision_recall_curve(y_test_bin, y_score)
                roc_dict[cls] = {"Model": model_name, "FPR": fpr, "TPR": tpr, "AUC": auc(fpr, tpr)}
                pr_dict[cls]  = {"Model": model_name, "Precision": prec, "Recall": rec, "AUC-PR": auc(rec, prec)}

                fpr_all.append(fpr)
                tpr_all.append(tpr)
                prec_all.append(prec)
                rec_all.append(rec)

            # Macro-average (simple mean)
            roc_macro = np.mean([v["AUC"] for v in roc_dict.values()])
            pr_macro  = np.mean([v["AUC-PR"] for v in pr_dict.values()])

            # Micro-average: concatenate TPs/FPs
            all_fpr = np.unique(np.concatenate(fpr_all))
            tpr_micro = np.zeros_like(all_fpr)
            for i in range(len(classes)):
                tpr_micro += np.interp(all_fpr, fpr_all[i], tpr_all[i])
            tpr_micro /= len(classes)
            roc_micro = auc(all_fpr, tpr_micro)

            all_rec = np.unique(np.concatenate(rec_all))
            prec_micro = np.zeros_like(all_rec)
            for i in range(len(classes)):
                prec_micro += np.interp(all_rec, rec_all[i], prec_all[i])
            prec_micro /= len(classes)
            pr_micro = auc(all_rec, prec_micro)

        # ---------------- Metric Calculator ----------------
        def scores(y_true, y_pred):
            average = "binary" if len(np.unique(y_true)) == 2 else "macro"
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
                "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, average=average, zero_division=0),
                "ROC-AUC (macro)": roc_macro,
                "ROC-AUC (micro)": roc_micro,
                "PR-AUC (macro)": pr_macro,
                "PR-AUC (micro)": pr_micro
            }

        train_metrics = scores(y_train, y_train_pred)
        test_metrics  = scores(y_test, y_test_pred)

        # Combine train/test metrics
        metrics = {
            "Model": model_name,
            **{k: f"{test_metrics[k]:.3f} ({train_metrics[k]:.3f})" for k in test_metrics}
        }

        return metrics, roc_dict, pr_dict

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 PLOT CONFUSION MATRIX
    def _plot_confusion_matrix(self, model_name, model, X_test, y_test):
        """
        Plots a confusion matrix for a given model and test set.

        Parameters
        ----------
        model_name : str
            Name of the model to use in titles and filenames.
        model : object
            Trained scikit-learn compatible model with a `predict` method.
        X_test : array-like of shape (n_samples, n_features)
            Test input features.
        y_test : array-like of shape (n_samples,)
            True labels for test data.

        Notes
        -----
        - Uses model.classes_ if available; otherwise falls back to unique labels in y_test.
        - Saves the figure as a high-resolution PNG in `self.output_dir/confusion_matrices`.
        - Uses a blue color map with integer annotations.
        """
        try:
            print(f"      └── Generating confusion matrices...")

            # Prepare output folder
            output_dir = os.path.join(self.output_dir, "confusion_matrices")
            os.makedirs(output_dir, exist_ok=True)

            # Predict
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            # Determine class labels
            if hasattr(model, "classes_"):
                class_labels = model.classes_
            else:
                class_labels = np.unique(y_test)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar=True,
                square=True,
                linewidths=0.5,
                ax=ax
            )
            ax.set_xlabel("Predicted Label", fontsize=14)
            ax.set_ylabel("True Label", fontsize=14)
            ax.set_title(f"{model_name} Confusion Matrix", fontsize=16)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save results
            os.makedirs(f"{self.output_dir}/confusion_matrices", exist_ok=True)
            file_path = f"{self.output_dir}/confusion_matrices/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(file_path, dpi=300)
            plt.close()

            rprint(f"      └── Saved confusion matrices importance to '{file_path}'")

        except Exception as e:
            print(f"      └── ⚠️  Failed to plot confusion matrix for {model_name}: {e}")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 PLOT CALIBRATION CURVES
    def _plot_calibration_curves(self, model_name, model, X_test, y_test):
        """
        Plot calibration curves for a given model.
        
        Handles both binary and multiclass classification. Saves figures to output directory.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model : object
            Trained classifier with a `predict_proba` method.
        X_test : array-like
            Test features.
        y_test : array-like
            True labels for test set.

        Returns
        -------
        None
        """
        try:
            print(f"      └── Generating calibration curves...")

            os.makedirs(f"{self.output_dir}/calibration_curves", exist_ok=True)
            file_path = f"{self.output_dir}/calibration_curves/calibration_curve_{model_name.replace(' ', '_').lower()}.png"

            if not hasattr(model, "predict_proba"):
                print(f"      └── ⚠️  Model {model_name} does not support `predict_proba`, skipping calibration curve.")
                return

            classes = np.unique(y_test)
            n_classes = len(classes)

            # Binary classification
            if n_classes == 2:
                prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:,1], n_bins=10)
                plt.figure(figsize=(8,6))
                plt.plot(prob_pred, prob_true, "s-", label="Calibration")
                plt.plot([0,1],[0,1], "k--", label="Perfectly calibrated")
                plt.title(f"{model_name} Calibration Curve", fontsize=16)
                plt.xlabel("Predicted Probability", fontsize=14)
                plt.ylabel("True Probability", fontsize=14)
                plt.ylim([0, 1.05])
                plt.legend()
                plt.tight_layout()
                plt.savefig(file_path, dpi=300)
                plt.close()

            # Multiclass classification (One-vs-Rest)
            else:
                y_test_probs = model.predict_proba(X_test)
                plt.figure(figsize=(10,7))
                for i, cls in enumerate(classes):
                    prob_true, prob_pred = calibration_curve((y_test==cls).astype(int), y_test_probs[:,i], n_bins=10)
                    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', linewidth=2, label=f"Class {cls}")
                plt.plot([0,1],[0,1], "k--", lw=1, label="Perfectly calibrated")
                plt.title(f"{model_name} Calibration Curves (Multiclass)", fontsize=16)
                plt.xlabel("Predicted Probability", fontsize=14)
                plt.ylabel("True Probability", fontsize=14)
                plt.ylim([0, 1.05])
                plt.legend(title="Classes", bbox_to_anchor=(1.05,1), loc="upper left")
                plt.tight_layout()
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()
                
            rprint(f"      └── Saved confusion matrices importance to '{file_path}'")

        except Exception as e:
            print(f"      └── ⚠️  Failed calibration curve for {model_name}: {e}")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 PLOT ROC CURVES
    def _plot_roc_curves(self, roc_data):
        """
        Plots ROC curves for multiple models:
        - Binary models → combined chart
        - Multiclass models → 
            * One combined macro-average chart (all models)
            * One combined micro-average chart (all models)
            * Per-class chart per model

        Parameters
        ----------
        roc_data : list
            List of dicts containing ROC info per model. Can handle multiclass as dict of classes.
        """
        print(f"\n   └── Generating ROC curves...")

        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("tab10")

        # Create output folder
        output_dir = os.path.join(self.output_dir, "roc_curves")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare figures and axes
        binary_fig, binary_ax = plt.subplots(figsize=(10, 7))
        macro_fig, macro_ax = plt.subplots(figsize=(10, 7))
        micro_fig, micro_ax = plt.subplots(figsize=(10, 7))

        for idx, d in enumerate(roc_data):
            if d is None:
                continue

            # Extract model name
            if isinstance(d, dict) and all(isinstance(v, dict) for v in d.values()):
                first_cls = next(iter(d))
                model_name = d[first_cls]["Model"]
            else:
                model_name = d["Model"]

            # --- Multiclass case ---
            if isinstance(d, dict) and all(isinstance(v, dict) for v in d.values()):
                # Macro-average
                all_fpr = np.unique(np.concatenate([v["FPR"] for v in d.values()]))
                mean_tpr = np.zeros_like(all_fpr)
                for cls_dict in d.values():
                    mean_tpr += np.interp(all_fpr, cls_dict["FPR"], cls_dict["TPR"])
                mean_tpr /= len(d)
                auc_macro = np.mean([v["AUC"] for v in d.values()])
                macro_ax.plot(all_fpr, mean_tpr, 
                            label=f"{model_name} (AUC={auc_macro:.3f})",
                            color=palette[idx % len(palette)],
                            linewidth=2)

                # Micro-average
                all_fpr_micro = np.unique(np.concatenate([v["FPR"] for v in d.values()]))
                tpr_micro = np.zeros_like(all_fpr_micro)
                for cls_dict in d.values():
                    tpr_micro += np.interp(all_fpr_micro, cls_dict["FPR"], cls_dict["TPR"])
                tpr_micro /= len(d)
                auc_micro = np.mean([v["AUC"] for v in d.values()])
                micro_ax.plot(all_fpr_micro, tpr_micro, 
                            label=f"{model_name} (AUC={auc_micro:.3f})",
                            color=palette[idx % len(palette)],
                            linewidth=2,
                            linestyle='--')

                # Per-class ROC
                per_class_fig, per_class_ax = plt.subplots(figsize=(10, 7))
                for cls, cls_dict in d.items():
                    per_class_ax.plot(cls_dict["FPR"], cls_dict["TPR"],
                                    label=f"Class {cls} (AUC={cls_dict['AUC']:.3f})",
                                    linewidth=2)
                per_class_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
                per_class_ax.set_title(f"Per-Class ROC Curves: {model_name}", fontsize=16)
                per_class_ax.set_xlabel("False Positive Rate", fontsize=14)
                per_class_ax.set_ylabel("True Positive Rate", fontsize=14)
                per_class_ax.set_xlim([0, 1])
                # Dynamic Y-limit: top 5% above max TPR
                per_class_ax.set_ylim([0, min(1.05, per_class_ax.get_ylim()[1]*1.05)])
                per_class_ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
                per_class_fig.tight_layout()
                per_class_fig.savefig(
                    os.path.join(output_dir, f"roc_{model_name.replace(' ', '_').lower()}.png"),
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close(per_class_fig)

            else:
                # --- Binary case ---
                binary_ax.plot(d["FPR"], d["TPR"], 
                           label=f"{model_name} (AUC={d['AUC']:.3f})",
                           color=palette[idx % len(palette)],
                           linewidth=2.5)

        # --- Finalize Binary ROC ---
        if binary_ax.has_data():
            binary_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
            binary_ax.set_xlabel("False Positive Rate", fontsize=14)
            binary_ax.set_ylabel("True Positive Rate", fontsize=14)
            binary_ax.set_title("ROC Curves (Binary Models)", fontsize=16)
            binary_ax.set_xlim([0, 1])
            # Dynamic Y-limit
            ymax = min(1.05, max([line.get_ydata().max() for line in binary_ax.lines])*1.05)
            binary_ax.set_ylim([0, ymax])
            binary_ax.legend()
            binary_fig.tight_layout()
            binary_fig.savefig(os.path.join(output_dir, "roc_combined_binary.png"), dpi=300)
            plt.close(binary_fig)

        # --- Finalize Macro ROC ---
        if macro_ax.has_data():
            macro_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
            macro_ax.set_xlabel("False Positive Rate", fontsize=14)
            macro_ax.set_ylabel("True Positive Rate", fontsize=14)
            macro_ax.set_title("ROC Curves (Macro Averages)", fontsize=16)
            macro_ax.set_xlim([0, 1])
            ymax = min(1.05, max([line.get_ydata().max() for line in macro_ax.lines])*1.05)
            macro_ax.set_ylim([0, ymax])
            macro_ax.legend()
            macro_fig.tight_layout()
            macro_fig.savefig(os.path.join(output_dir, "roc_combined_macro.png"), dpi=300)
            plt.close(macro_fig)

        # --- Finalize Micro ROC ---
        if micro_ax.has_data():
            micro_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
            micro_ax.set_xlabel("False Positive Rate", fontsize=14)
            micro_ax.set_ylabel("True Positive Rate", fontsize=14)
            micro_ax.set_title("ROC Curves (Micro Averages)", fontsize=16)
            micro_ax.set_xlim([0, 1])
            ymax = min(1.05, max([line.get_ydata().max() for line in micro_ax.lines])*1.05)
            micro_ax.set_ylim([0, ymax])
            micro_ax.legend()
            micro_fig.tight_layout()
            micro_fig.savefig(os.path.join(output_dir, "roc_combined_micro.png"), dpi=300)
            plt.close(micro_fig)

        rprint(f"   └── Saved ROC curves to 'output/roc_curves' folder")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📀 PLOT PRECISION-RECALL CURVES
    def _plot_pr_curves(self, pr_data):
        """
        Plots Precision-Recall curves for multiple models:
        - Binary models → combined chart
        - Multiclass models → 
            * One combined macro-average chart (all models)
            * One combined micro-average chart (all models)
            * Per-class chart per model

        Parameters
        ----------
        pr_data : list
            List of dicts containing PR info per model. Can handle multiclass as dict of classes.
        """
        print(f"   └── Generating PR curves...")

        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("tab10")

        # Create output folder
        output_dir = os.path.join(self.output_dir, "pr_curves")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare figures
        binary_fig, binary_ax = plt.subplots(figsize=(10, 7))
        macro_fig, macro_ax = plt.subplots(figsize=(10, 7))
        micro_fig, micro_ax = plt.subplots(figsize=(10, 7))

        for idx, d in enumerate(pr_data):
            if d is None:
                continue

            # Extract model name
            if isinstance(d, dict) and all(isinstance(v, dict) for v in d.values()):
                first_cls = next(iter(d))
                model_name = d[first_cls]["Model"]
            else:
                model_name = d["Model"]

            # --- Multiclass case ---
            if isinstance(d, dict) and all(isinstance(v, dict) for v in d.values()):
                # Macro-average
                all_rec = np.unique(np.concatenate([v["Recall"] for v in d.values()]))
                mean_prec = np.zeros_like(all_rec)
                for cls_dict in d.values():
                    mean_prec += np.interp(all_rec, cls_dict["Recall"][::-1], cls_dict["Precision"][::-1])[::-1]
                mean_prec /= len(d)
                auc_macro = np.mean([v["AUC-PR"] for v in d.values()])
                macro_ax.plot(all_rec, mean_prec,
                            label=f"{model_name} (Macro AUC-PR={auc_macro:.3f})",
                            color=palette[idx % len(palette)],
                            linewidth=2.5)

                # Micro-average
                all_rec_micro = np.unique(np.concatenate([v["Recall"] for v in d.values()]))
                mean_prec_micro = np.zeros_like(all_rec_micro)
                for cls_dict in d.values():
                    mean_prec_micro += np.interp(all_rec_micro, cls_dict["Recall"][::-1], cls_dict["Precision"][::-1])[::-1]
                mean_prec_micro /= len(d)
                auc_micro = np.mean([v["AUC-PR"] for v in d.values()])
                micro_ax.plot(all_rec_micro, mean_prec_micro,
                            label=f"{model_name} (Micro AUC-PR={auc_micro:.3f})",
                            color=palette[idx % len(palette)],
                            linewidth=2.5,
                            linestyle='--')

                # Per-class PR
                per_class_fig, per_class_ax = plt.subplots(figsize=(10, 7))
                for cls, cls_dict in d.items():
                    per_class_ax.plot(cls_dict["Recall"], cls_dict["Precision"],
                                    label=f"Class {cls} (AUC-PR={cls_dict['AUC-PR']:.3f})",
                                    linewidth=2)
                per_class_ax.set_title(f"Per-Class PR Curves: {model_name}", fontsize=16)
                per_class_ax.set_xlabel("Recall", fontsize=14)
                per_class_ax.set_ylabel("Precision", fontsize=14)
                per_class_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
                per_class_ax.set_xlim([0, 1])
                # Dynamic Y-limit with 5% margin
                ymax = min(1.05, max([line.get_ydata().max() for line in per_class_ax.lines])*1.05)
                per_class_ax.set_ylim([0, ymax])
                per_class_ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
                per_class_fig.tight_layout()
                per_class_fig.savefig(
                    os.path.join(output_dir, f"pr_{model_name.replace(' ', '_').lower()}.png"),
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close(per_class_fig)

            else:
                # --- Binary case ---
                binary_ax.plot(d["Recall"], d["Precision"],
                            label=f"{model_name} (AUC-PR={d['AUC-PR']:.3f})",
                            color=palette[idx % len(palette)],
                            linewidth=2.5)

        # --- Finalize Binary PR ---
        if binary_ax.has_data():
            binary_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
            binary_ax.set_xlabel("Recall", fontsize=14)
            binary_ax.set_ylabel("Precision", fontsize=14)
            binary_ax.set_title("Precision-Recall Curves (Binary Models)", fontsize=16)
            binary_ax.set_xlim([0, 1])
            ymax = min(1.05, max([line.get_ydata().max() for line in binary_ax.lines])*1.05)
            binary_ax.set_ylim([0, ymax])
            binary_ax.legend()
            binary_fig.tight_layout()
            binary_fig.savefig(os.path.join(output_dir, "pr_combined_binary.png"), dpi=300)
            plt.close(binary_fig)

        # --- Finalize Macro PR ---
        if macro_ax.has_data():
            macro_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
            macro_ax.set_xlabel("Recall", fontsize=14)
            macro_ax.set_ylabel("Precision", fontsize=14)
            macro_ax.set_title("Precision-Recall Curves (Macro Averages)", fontsize=16)
            macro_ax.set_xlim([0, 1])
            ymax = min(1.05, max([line.get_ydata().max() for line in macro_ax.lines])*1.05)
            macro_ax.set_ylim([0, ymax])
            macro_ax.legend()
            macro_fig.tight_layout()
            macro_fig.savefig(os.path.join(output_dir, "pr_combined_macro.png"), dpi=300)
            plt.close(macro_fig)

        # --- Finalize Micro PR ---
        if micro_ax.has_data():
            micro_ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
            micro_ax.set_xlabel("Recall", fontsize=14)
            micro_ax.set_ylabel("Precision", fontsize=14)
            micro_ax.set_title("Precision-Recall Curves (Micro Averages)", fontsize=16)
            micro_ax.set_xlim([0, 1])
            ymax = min(1.05, max([line.get_ydata().max() for line in micro_ax.lines])*1.05)
            micro_ax.set_ylim([0, ymax])
            micro_ax.legend()
            micro_fig.tight_layout()
            micro_fig.savefig(os.path.join(output_dir, "pr_combined_micro.png"), dpi=300)
            plt.close(micro_fig)

        rprint(f"   └── Saved PR curves to 'output/pr_curves' folder")

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
        print(f"      └── Generating feature importance scores...")

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

            rprint(f"      └── Saved feature importance to '{file_path}'")

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
            print(f"      └── Generating learning curve...")

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

            rprint(f"      └── Saved learning curve to '{file_path}'")

        except Exception as e:
            print(f"      └── ⚠️ Failed to generate learning curve: {e}")

    ########################################################################################################################################
    ########################################################################################################################################
    # 📝 SAVE EVALUATION METRICS
    def _save_evaluation_metrics(self, results, filename="evaluation_metrics_summary.md"):
        """
        Save a consolidated evaluation metrics summary for all models in a professional, 
        Markdown/README-friendly format.

        Parameters
        ----------
        results : list of dict
            Each dict contains metrics for one model.
        filename : str, default "evaluation_metrics_summary.md"
            Output filename.

        Returns
        -------
        None
            Saves metrics to a Markdown file in the output directory.
        """
        try:
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# 📊 Model Evaluation Metrics Summary\n\n")
                f.write("This document summarizes the performance of all trained models.\n\n")
                f.write("**Note:** Test metrics are reported first; training metrics are shown in `(brackets)`.\n\n")

                # Convert results to DataFrame for tabulation
                df = pd.DataFrame(results)

                # Convert metrics to Markdown table using tabulate
                metrics_table = tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f")
                f.write(metrics_table + "\n\n")

                f.write("---\n")
                f.write("Generated automatically by the Model Evaluation pipeline.\n")

                # VS Code preview refresh tip
                f.write("> ⚠️ **Tip:** If you are viewing this in VS Code Markdown Preview, ")
                f.write("you may need to manually refresh (`Ctrl+Shift+R`) or re-open the preview ")
                f.write("to see the latest updates.\n")

            rprint(f"\n   └── Saved evaluation metrics summary to '{output_path}'")

        except Exception as e:
            print(f"   └── ⚠️ Failed to save evaluation metrics: {e}")