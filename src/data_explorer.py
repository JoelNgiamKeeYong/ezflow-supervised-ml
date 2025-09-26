# ===========================================================================================================================================
# 🔎 DATA EXPLORER CLASS
# ===========================================================================================================================================

import re
import dtale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore
from IPython.display import display
import squarify
from scipy.stats import chi2_contingency, pearsonr, spearmanr, pointbiserialr, f_oneway
from typing import List
from sklearn.linear_model import LinearRegression
from rich import print as rprint

class DataExplorer:
    """
    Class for exploratory data analysis (EDA). Methods are stateless.
    All analysis functions take a DataFrame as input.
    """

    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 PERFORM UNIVARIATE ANALYSIS
    @staticmethod
    def perform_univariate_analysis(
        df: pd.DataFrame,
        feature: str,
        show_plots: bool = True,
        top_n_pie: int = 5,
        skew_thresh: float = 1.0,
        kurt_thresh: float = 3.0,
        high_card_threshold: int = 25,
        rare_threshold: float = 0.01,
        bins: int = 30
    ):
        """
        Perform univariate analysis on a single feature (numerical or categorical) in a DataFrame.

        This method automatically detects the feature type and calls the appropriate analysis:
        - Numerical: Computes summary statistics, skewness, kurtosis, outlier detection, missing values, and plots.
        - Categorical: Computes counts, percentages, frequency tables, warnings for high cardinality or rare categories, missing values, and plots.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the feature to analyze.
        feature : str
            The column name of the feature to analyze.
        show_plots : bool, default=True
            Whether to generate visualizations for the feature.
        top_n_pie : int, default=5
            For categorical features: number of top categories to display individually in the pie chart (remaining grouped as "Others").
        skew_thresh : float, default=1.0
            Threshold for skewness above which a warning is shown for numerical features.
        kurt_thresh : float, default=3.0
            Threshold for kurtosis above which a warning is shown for numerical features.
        high_card_threshold : int, default=25
            Threshold for high cardinality in categorical features to trigger a warning.
        rare_threshold : float, default=0.01
            Threshold for rare categories (fraction of total) in categorical features to trigger a warning.
        bins : int, default=30
            Number of bins to use for numerical feature histograms.

        Returns
        -------
        None
            Prints analysis results to the console and displays plots if `show_plots=True`.

        Notes
        -----
        - Automatically selects the type of analysis based on the feature's data type.
        - For numerical features, outlier detection uses either:
            - IQR Method (1.5 × IQR) for skewed distributions, or
            - Z-Score Method (|z| > 3) for near-normal distributions.
        - Skewness and kurtosis warnings indicate when values exceed the specified thresholds.
        - Categorical warnings include high cardinality, rare categories, constant values, and potential inconsistencies in capitalization or whitespace.
        """
        if feature not in df.columns:
            print(f"❌  Feature '{feature}' not found in the DataFrame.")
            return

        col_type = "categorical" if df[feature].dtype in ['object', 'category'] else "numerical"
        print(f"🔎 Univariate Analysis for '{feature}' (Type: {col_type})\n")

        if pd.api.types.is_numeric_dtype(df[feature]):
            # Numerical feature
            DataExplorer._analyse_num(
                df, feature, show_plots, bins, skew_thresh, kurt_thresh
            )
        elif pd.api.types.is_categorical_dtype(df[feature]) or df[feature].dtype == 'object':
            # Categorical feature
            DataExplorer._analyse_cat(
                df, feature, show_plots, top_n_pie, high_card_threshold, rare_threshold
            )
        else:
            print(f"❌  '{feature}' is neither numerical nor categorical. Data type: {df[feature].dtype}")

    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 ANALYSE NUMERICAL
    @staticmethod
    def _analyse_num(df, feature, show_plots, bins, skew_thresh, kurt_thresh):
        """
        Perform a comprehensive univariate analysis on a numerical feature.

        This function provides an overview of a numerical column by:
        1. Displaying data type and number of unique non-NA values.
        2. Showing summary statistics (count, mean, std, min, quartiles, max).
        3. Calculating skewness and kurtosis, with a warning if they exceed thresholds.
        4. Detecting outliers using either:
        - IQR Method (for skewed distributions), or
        - Z-Score Method (for approximately normal distributions).
        It also prints the method used and the corresponding bounds.
        5. Reporting missing values and showing a sample of rows with missing data.
        6. Optionally plotting visualizations:
        - Histogram with KDE
        - Boxplot
        - Violin plot
        - QQ plot for normality assessment

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the feature to analyze.
        feature : str
            The column name of the numerical feature to analyze.
        show_plots : bool, default=True
            Whether to generate visual plots for the feature.
        bins : int, default=30
            Number of bins to use for the histogram.
        skew_thresh : float, default=1.0
            Threshold for skewness above which a warning is displayed.
        kurt_thresh : float, default=3.0
            Threshold for kurtosis above which a warning is displayed.

        Returns
        -------
        None
            Prints analysis results and shows plots if requested.

        Notes
        -----
        - Outlier detection method is chosen automatically based on skewness and kurtosis:
            - IQR method (1.5 × IQR) is used for skewed distributions.
            - Z-Score method (|z| > 3) is used for near-normal distributions.
        - Skewness and kurtosis warnings indicate values above the defined thresholds.
        """

        # Data type and unique values
        feature_series = df[feature].dropna()
        print(f"📘 Data Type: {feature_series.dtype}")
        print(f"💎 Unique Non-NA: {feature_series.nunique()}")

        # Summary statistics
        stats = feature_series.describe()
        print("📊 Summary Statistics:")
        display(stats.to_frame().T.style.format("{:.2f}"))

        # Skewness and kurtosis
        skew, kurt = feature_series.skew(), feature_series.kurtosis()
        skew_msg = f"📈 Skewness: {skew:.2f}"
        if abs(skew) > skew_thresh:
            skew_msg += f" ⚠️ (> {skew_thresh})"
        kurt_msg = f"📈 Kurtosis: {kurt:.2f}"
        if kurt > kurt_thresh:
            kurt_msg += f" ⚠️ (> {kurt_thresh})"
        print(skew_msg)
        print(kurt_msg)

        # Detect outliers
        if abs(skew) > skew_thresh or abs(kurt) > 2:
            # Skewed distribution → IQR Method
            method = "IQR Method (1.5 × IQR)"
            Q1, Q3 = df[feature].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            bounds_msg = f"   └── Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
        else:
            # Relatively normal distribution → Z-Score Method
            method = "Z-Score Method (|z| > 3)"
            z_scores = zscore(df[feature].dropna())
            outliers = df[abs(z_scores) > 3]
            mean = df[feature].mean()
            std = df[feature].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            bounds_msg = f"   └── Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"

        print(f"\n🔍 Outlier Detection: {method}")
        print(bounds_msg)
        if outliers.empty:
            print("   └── ✅ No outliers detected.")
        else:
            print(f"   └── ⚠️ {len(outliers)} outliers found ({len(outliers)/len(df)*100:.2f}% of rows)")

        # Missing values
        missing_count = df[feature].isnull().sum()
        if missing_count == 0:
            print("\n✅ No rows with missing values found.")
        else:
            total_rows = len(df)
            print(f"\n⚠️  Missing values: {missing_count}/{total_rows} ({missing_count/total_rows*100:.2f}%)")
            display(df[df[feature].isnull()].head())

        # Plot visualizations
        if show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            # Histogram
            sns.histplot(df[feature], bins=bins, kde=True, ax=axes[0, 0], color="#4C72B0")
            axes[0, 0].set_title("Histogram + KDE")

            # Boxplot
            sns.boxplot(y=df[feature], ax=axes[0, 1], color='#C44E52')
            axes[0, 1].set_title("Boxplot")

            # Violin plot
            sns.violinplot(y=df[feature], ax=axes[1, 0], color='#55A868')
            axes[1, 0].set_title("Violin Plot")

            # QQ Plot
            sm.qqplot(df[feature].dropna(), line='s', ax=axes[1, 1], markerfacecolor='#FFA500', markeredgecolor='black')
            axes[1, 1].set_title("QQ Plot")

            fig.suptitle(f"Univariate Analysis of '{feature}'", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()

    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 ANALYSE CATEGORICAL
    @staticmethod
    def _analyse_cat(
        df: pd.DataFrame,
        feature: str,
        show_plots: bool = True,
        top_n_pie: int = 5,
        high_card_threshold: int = 25,
        rare_threshold: float = 0.01
    ):
        """
        Perform a comprehensive univariate analysis on a categorical feature.

        This function provides an overview of a categorical column by:
        1. Displaying data type and number of unique non-NA values.
        2. Listing all unique values.
        3. Showing a frequency table with counts and percentages.
        4. Reporting warnings:
            - Constant value
            - High cardinality
            - Rare categories
            - Whitespace / capitalization issues
        5. Reporting missing values and showing a sample of rows with missing data.
        6. Optionally plotting visualizations:
            - Countplot
            - Pie chart for top categories

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the feature to analyze.
        feature : str
            The column name of the categorical feature to analyze.
        show_plots : bool, default=True
            Whether to generate visual plots for the feature.
        top_n_pie : int, default=5
            Number of top categories to display individually in the pie chart; remaining grouped as "Others".
        high_card_threshold : int, default=25
            Threshold for high cardinality to trigger a warning.
        rare_threshold : float, default=0.01
            Threshold for rare categories (fraction of total) to trigger a warning.

        Returns
        -------
        None
            Prints analysis results and shows plots if requested.
        """
        feature_series = df[feature].dropna()
        print(f"📘 Data Type: {feature_series.dtype}")
        print(f"💎 Unique Non-NA Values: {feature_series.nunique()}")
        print(f"📋 Unique Values List: {list(feature_series.unique())}")

        # Frequency table
        counts = feature_series.value_counts()
        percentages = feature_series.value_counts(normalize=True) * 100
        freq_table = pd.DataFrame({
            'Count': counts.apply(lambda x: f"{x:,}"),
            'Percentage (%)': percentages.round(2)
        })
        freq_table.index.name = None
        print("📊 Frequency Table:")
        display(freq_table)

        # Warnings
        if feature_series.nunique() == 1:
            print(f"⚠️  Constant Value: Feature is constant with value: {feature_series.unique()[0]}")
        elif feature_series.nunique() > high_card_threshold:
            print(f"⚠️  High Cardinality (>{high_card_threshold}): {feature_series.nunique()} unique categories")

        rare_cats = (counts / counts.sum() < rare_threshold)
        if rare_cats.any():
            print(f"⚠️  Rare Categories (<{rare_threshold*100:.0f}%): {rare_cats.sum()} found")

        stripped = feature_series.astype(str).str.strip()
        if not feature_series.astype(str).equals(stripped):
            print("⚠️  Detected leading/trailing whitespace.")
        lowercase = feature_series.astype(str).str.lower()
        if lowercase.nunique() < feature_series.nunique():
            print("⚠️  Potential inconsistent capitalization.")

        # Missing values
        missing_count = df[feature].isnull().sum()
        if missing_count == 0:
            print("\n✅ No rows with missing values found.")
        else:
            total_rows = len(df)
            print(f"\n⚠️  Missing values: {missing_count}/{total_rows} ({missing_count/total_rows*100:.2f}%)")
            display(df[df[feature].isnull()].head())

        # Plots
        if show_plots:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

            # Countplot
            sns.countplot(data=df, x=feature, order=counts.index, ax=axes[0])
            axes[0].set_title("Countplot")
            axes[0].tick_params(axis='x', rotation=45)

            # Pie chart
            if len(counts) > top_n_pie:
                top_n = counts[:top_n_pie]
                others = pd.Series(counts[top_n_pie:].sum(), index=["Others"])
                pie_data = pd.concat([top_n, others])
            else:
                pie_data = counts

            axes[1].pie(
                pie_data,
                labels=pie_data.index,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100 * pie_data.sum())})',
                startangle=90
            )
            axes[1].set_title("Pie Chart")

            fig.suptitle(f"Univariate Analysis of '{feature}'", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()
    
    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 PERFORM BIVARIATE ANALYSIS
    @staticmethod
    def perform_bivariate_analysis(df: pd.DataFrame, col1: str, col2: str, show_plots: bool = True):
        """
        Determine column types and delegate to appropriate analysis method.
        """
        if col1 not in df.columns:
            print(f"❌ Column '{col1}' not found in the DataFrame.")
            return
        if col2 not in df.columns:
            print(f"❌ Column '{col2}' not found in the DataFrame.")
            return

        col1_type = "categorical" if df[col1].dtype in ['object', 'category'] else "numerical"
        col2_type = "categorical" if df[col2].dtype in ['object', 'category'] else "numerical"
        print(f"🔎 Bivariate Analysis: '{col1}' ({col1_type}) vs. '{col2}' ({col2_type})\n")

        if col1_type == "numerical" and col2_type == "numerical":
            DataExplorer._analyse_num_num(df, col1, col2, show_plots)
        elif (col1_type == "numerical" and col2_type == "categorical") or (col1_type == "categorical" and col2_type == "numerical"):
            DataExplorer._analyse_num_cat(df, col1, col2, show_plots)
        elif col1_type == "categorical" and col2_type == "categorical":
            DataExplorer._analyse_cat_cat(df, col1, col2, show_plots)
        else:
            print(f"❌  Cannot analyze combination: {col1_type} & {col2_type}")


    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 NUMERICAL x NUMERICAL ANALYSIS
    @staticmethod
    def _analyse_num_num(df, col1, col2, show_plots):
        pair_df = df[[col1, col2]].dropna()

        # Pearson Correlation
        pearson_corr, pearson_p = pearsonr(pair_df[col1], pair_df[col2])
        print(f"🧪 Pearson Correlation: {pearson_corr:.2f}, p-value: {pearson_p:.4f}")
        if pearson_p < 0.05:
            if pearson_corr < -0.7:
                print("   └── ⚠️ Significant correlation with very strong negative effect.")
            elif pearson_corr < -0.5:
                print("   └── ⚠️ Significant correlation with strong negative effect.")
            elif pearson_corr < -0.3:
                print("   └── ⚠️ Significant correlation with moderate negative effect.")
            elif pearson_corr < 0.3:
                print("   └── Significant correlation with weak effect.")
            elif pearson_corr < 0.5:
                print("   └── ⚠️ Significant correlation with moderate positive effect.")
            elif pearson_corr < 0.7:
                print("   └── ⚠️ Significant correlation with strong positive effect.")
            else:
                print("   └── ⚠️ Significant correlation with very strong positive effect.")
        else:
            print("   └── No significant linear correlation (Pearson).")

        # Spearman Correlation
        spearman_corr, spearman_p = spearmanr(pair_df[col1], pair_df[col2])
        print(f"\n🧪 Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_p:.4f}")
        if spearman_p < 0.05:
            if spearman_corr < -0.7:
                print("   └── ⚠️ Significant monotonic correlation with very strong negative effect.")
            elif spearman_corr < -0.5:
                print("   └── ⚠️ Significant monotonic correlation with strong negative effect.")
            elif spearman_corr < -0.3:
                print("   └── ⚠️ Significant monotonic correlation with moderate negative effect.")
            elif spearman_corr < 0.3:
                print("   └── Significant monotonic correlation with weak effect.")
            elif spearman_corr < 0.5:
                print("   └── ⚠️ Significant monotonic correlation with moderate positive effect.")
            elif spearman_corr < 0.7:
                print("   └── ⚠️ Significant monotonic correlation with strong positive effect.")
            else:
                print("   └── ⚠️ Significant monotonic correlation with very strong positive effect.")
        else:
            print("   └── No significant monotonic correlation (Spearman).")

        # Plots
        if show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            # Scatter + regression
            sns.regplot(data=pair_df, x=col1, y=col2, scatter_kws={'alpha':0.6, 'color':'#4A90E2'}, line_kws={'color':'#E94E77'}, ax=axes[0,0])
            axes[0,0].set_title("Scatter Plot with Regression Line")
            # Hexbin
            axes[0,1].hexbin(pair_df[col1], pair_df[col2], gridsize=30, cmap='viridis')
            axes[0,1].set_title("Hexbin Plot: Density")
            # Correlation heatmap
            sns.heatmap(pair_df[[col1,col2]].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=axes[1,0])
            axes[1,0].set_title("Correlation Heatmap")
            # Residual plot
            model = LinearRegression().fit(pair_df[[col1]], pair_df[col2])
            residuals = pair_df[col2] - model.predict(pair_df[[col1]])
            sns.scatterplot(x=model.predict(pair_df[[col1]]), y=residuals, ax=axes[1,1], alpha=0.6, color='#50E3C2')
            axes[1,1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1,1].set_title("Residual Plot")
            axes[1,1].set_xlabel("Predicted Values")
            axes[1,1].set_ylabel("Residuals")

            fig.suptitle(f"Bivariate Analysis of '{col1}' and '{col2}'", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()

    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 NUMERICAL x CATEGORICAL ANALYSIS
    @staticmethod
    def _analyse_num_cat(df, col1, col2, show_plots):
        # Assign numerical & categorical
        if pd.api.types.is_numeric_dtype(df[col1]):
            num_col, cat_col = col1, col2
        else:
            num_col, cat_col = col2, col1

        pair_df = df[[num_col, cat_col]].dropna()

        # Summarize numeric by category
        print("📑 Summary Statistics by Category:")
        grouped = pair_df.groupby(cat_col)[num_col].agg(['mean','median','std']).reset_index()
        grouped[['mean', 'median', 'std']] = grouped[['mean', 'median', 'std']].astype(float)
        display(grouped.style.format({
            'mean': '{:,.2f}',
            'median': '{:,.2f}',
            'std': '{:,.2f}'
        }))

        # Statistical Tests
        if pair_df[cat_col].nunique() == 2:
            r_val, p_val = pointbiserialr(pair_df[cat_col].astype('category').cat.codes, pair_df[num_col])
            print(f"🧪 Point-biserial correlation: {r_val:.2f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                effect = abs(r_val)
                if effect > 0.7:
                    print("   └── ⚠️ Significant correlation with very strong effect.")
                elif effect > 0.5:
                    print("   └── ⚠️ Significant correlation with strong effect.")
                elif effect > 0.3:
                    print("   └── ⚠️ Significant correlation with moderate effect.")
                else:
                    print("   └── Significant correlation with weak effect.")
            else:
                print("   └── No significant correlation.")
        else:
            groups = [g[num_col].values for _, g in pair_df.groupby(cat_col)]
            f_stat, p_val = f_oneway(*groups)
            print(f"🧪 ANOVA F-statistic: {f_stat:.2f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                if f_stat > 10:
                    print("   └── ⚠️ Significant differences with very strong evidence.")
                elif f_stat > 5:
                    print("   └── ⚠️ Significant differences with strong evidence.")
                elif f_stat > 2:
                    print("   └── ⚠️ Significant differences with moderate evidence.")
                else:
                    print("   └── Significant differences with weak evidence.")
            else:
                print("   └── No significant differences across categories.")

        # Plots
        if show_plots:
            fig, axes = plt.subplots(2,2,figsize=(14,8))
            sns.boxplot(data=pair_df, x=cat_col, y=num_col, hue=cat_col, ax=axes[0,0], palette='viridis', dodge=False, legend=False)
            axes[0,0].set_title("Boxplot")
            sns.violinplot(data=pair_df, x=cat_col, y=num_col, hue=cat_col, ax=axes[0,1], palette='viridis', dodge=False, legend=False)
            axes[0,1].set_title("Violin Plot")
            sns.barplot(data=grouped, x=cat_col, y='mean', hue=cat_col, ax=axes[1,0], palette='viridis', dodge=False, legend=False)
            axes[1,0].set_title("Bar Plot of Means")
            sns.stripplot(data=pair_df, x=cat_col, y=num_col, ax=axes[1,1], color='green', jitter=True, alpha=0.6)
            axes[1,1].set_title("Scatter Plot with Jitter")

            fig.suptitle(f"Bivariate Analysis of '{col1}' and '{col2}'", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()

    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 CATEGORICAL x CATEGORICAL ANALYSIS
    @staticmethod
    def _analyse_cat_cat(df, col1, col2, show_plots):
        pair_df = df[[col1, col2]].dropna()
        crosstab_raw = pd.crosstab(pair_df[col1], pair_df[col2])
        crosstab_prop = pd.crosstab(pair_df[col1], pair_df[col2], normalize='index')

        # Chi-square test
        chi2, p, _, _ = chi2_contingency(crosstab_raw)
        print(f"🧪 Chi2 Statistic: {chi2:.2f}, p-value: {p:.4f}")
        n, r, c = pair_df.shape[0], crosstab_raw.shape[0], crosstab_raw.shape[1]
        cramers_v = np.sqrt(chi2 / (n * min(r-1,c-1)))
        print(f"🧪 Cramér's V: {cramers_v:.2f}")
        if p < 0.05:
            if cramers_v > 0.5:
                print("   └── ⚠️  Significant association with very strong effect.")
            elif cramers_v > 0.3:
                print("   └── ⚠️  Significant association with strong effect.")
            elif cramers_v > 0.1:
                print("   └── ⚠️  Significant association with moderate effect.")
            else:
                print("   └── Significant association with weak effect.")
        else:
            print("   └── ⚠️  No significant association.")

        # Plots
        if show_plots:
            fig, axes = plt.subplots(2,2,figsize=(14,8))
            sns.countplot(data=pair_df, x=col1, hue=col2, palette='tab10', ax=axes[0,0])
            axes[0,0].set_title("Countplot with Hue")
            # Treemap
            proportions = pair_df.groupby([col1,col2]).size().reset_index(name='counts')
            proportions['proportion'] = proportions['counts']/proportions['counts'].sum()
            axes[0,1].axis('off')
            squarify.plot(sizes=proportions['proportion'], label=proportions.apply(lambda x:f"{x[col1]}-{x[col2]}",axis=1), alpha=0.8, ax=axes[0,1])
            axes[0,1].set_title("Treemap")
            # Heatmap
            counts_str = crosstab_raw.apply(lambda col: col.map('{:,}'.format))
            percent_str = (crosstab_prop*100).round(1).astype(str)+'%'
            annot = counts_str + "\n(" + percent_str + ")"
            sns.heatmap(crosstab_prop, annot=annot, fmt='', cmap='coolwarm', cbar=False, ax=axes[1,0])
            axes[1,0].set_title("Heatmap")
            # Stacked bar chart
            crosstab_prop.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[1,1])
            axes[1,1].set_title("Stacked Bar Chart")

            fig.suptitle(f"Bivariate Analysis of '{col1}' and '{col2}'", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()

    ###################################################################################################################################
    ###################################################################################################################################
    # 🔍 COMPARE DATAFRAMES
    @staticmethod
    def compare_dataframes(
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        before_name: str = "before",
        after_name: str = "after"
    ) -> None:
        """
        Compare two DataFrames and print summary of changes in rows, columns, and memory usage.
        """
        # Rows
        rows_before = df_before.shape[0]
        rows_after = df_after.shape[0]
        rows_dropped = rows_before - rows_after

        # Columns
        cols_before = df_before.shape[1]
        cols_after = df_after.shape[1]
        dropped_columns = set(df_before.columns) - set(df_after.columns)
        added_columns = set(df_after.columns) - set(df_before.columns)

        def _normalize(col: str) -> str:
            return re.sub(r'[^a-z0-9]', '', col.lower().strip()) if col else ""
    
        renames = []
        for a in list(added_columns):
            norm_a = _normalize(a)
            matches = [b for b in dropped_columns if _normalize(b) == norm_a]
            if matches:
                renames.append((matches[0], a))
                dropped_columns.discard(matches[0]); added_columns.discard(a)

        # Memory usage
        size_before = df_before.memory_usage(deep=True).sum() / (1024 ** 2)
        size_after = df_after.memory_usage(deep=True).sum() / (1024 ** 2)
        size_change = size_after - size_before
        size_pct = (abs(size_change) / size_before * 100) if size_before > 0 else 0

        # Header
        rprint(f"\n[bold cyan]🔍 Comparing DataFrames[/bold cyan]: {before_name} → {after_name}")

        # Rows summary
        rprint(f"   └── [yellow]Rows[/yellow]: {rows_before:,} → {rows_after:,}", end="")
        if rows_dropped > 0:
            rprint(f"  |  Dropped: [red]{rows_dropped:,} ({rows_dropped/rows_before:.2%})[/red]")
        elif rows_dropped < 0:
            rprint(f"  |  Added: [green]{abs(rows_dropped):,} ({abs(rows_dropped)/rows_before:.2%})[/green]")
        else:
            rprint("  |  [cyan]No changes[/cyan]")

        # Columns summary
        if dropped_columns or added_columns:
            rprint(f"   └── [yellow]Columns[/yellow]: {cols_before} → {cols_after}")
            if dropped_columns:
                rprint(f"       └── Dropped: [red]{', '.join(sorted(dropped_columns))}[/red]")
            if added_columns:
                rprint(f"       └── Added: [green]{', '.join(sorted(added_columns))}[/green]")
        else:
            rprint(f"   └── [yellow]Columns[/yellow]: {cols_before} → {cols_after}  |  [cyan]No changes[/cyan]")

        # Memory summary
        rprint(f"   └── [yellow]Memory[/yellow]: {size_before:.2f} MB → {size_after:.2f} MB", end="")
        if size_change > 0:
            rprint(f"  |  Increase: [red]{size_change:.2f} MB ({size_pct:.2f}%)[/red]")
        elif size_change < 0:
            rprint(f"  |  Reduction: [green]{abs(size_change):.2f} MB ({size_pct:.2f}%)[/green]")
        else:
            rprint("  |  [cyan]No change[/cyan]")

###################################################################################################################################
###################################################################################################################################
# 🔍 LAUNCH D-TALE VISUALIZATION
    @staticmethod
    def show_dtale(
        df: pd.DataFrame,
        port: int = 5000,
        open_browser: bool = True,
        max_rows: int = 100,
        max_columns: int = None,
        float_format: str = "{:,.2f}",
        session_name: str = None
    ):
        """
        Launch D-Tale for interactive data exploration with optional formatting.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to visualize
        port : int, optional
            Port to run D-Tale on
        open_browser : bool, optional
            Open the browser automatically
        max_rows : int, optional
            Maximum number of rows displayed in D-Tale
        max_columns : int or None, optional
            Maximum number of columns displayed
        float_format : str, optional
            Formatting for float numbers
        session_name : str, optional
            Name for the D-Tale session (useful if multiple sessions)

        Returns
        -------
        dtale.DTale
            The D-Tale object
        """
        # Set pandas display options
        pd.set_option("display.max_rows", max_rows)
        if max_columns:
            pd.set_option("display.max_columns", max_columns)
        pd.set_option("display.width", 150)
        pd.set_option("display.float_format", float_format.format)

        rprint(f"\n[bold cyan]🔍 Launching D-Tale[/bold cyan] on port {port} with session name: {session_name or 'default'}")
        rprint(f"   └── DataFrame shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

        # Launch D-Tale
        d = dtale.show(
            df,
            port=port,
            subprocess=True,
            ignore_duplicate=True,
            name=session_name
        )

        if open_browser:
            d.open_browser()

        rprint(f"   └── [green]D-Tale launched successfully![/green]")
        return d