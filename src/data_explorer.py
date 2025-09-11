import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore
from IPython.display import display
from typing import List

class DataExplorer:
    """
    Class for exploratory data analysis (EDA). Methods are stateless.
    All analysis functions take a DataFrame as input.
    """

    # -----------------------------
    # Public Method
    # -----------------------------
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
        """Perform univariate analysis on a single column."""
        if feature not in df.columns:
            print(f"❌ Feature '{feature}' not found in the DataFrame.")
            return

        col_type = "categorical" if df[feature].dtype in ['object', 'category'] else "numerical"
        print(f"🔎 Univariate Analysis for '{feature}' (Type: {col_type})\n")

        if pd.api.types.is_numeric_dtype(df[feature]):
            DataExplorer._analyze_numerical(
                df, feature, show_plots, bins, skew_thresh, kurt_thresh
            )
        elif pd.api.types.is_categorical_dtype(df[feature]) or df[feature].dtype == 'object':
            DataExplorer._analyze_categorical(
                df, feature, show_plots, top_n_pie, high_card_threshold, rare_threshold
            )
        else:
            print(f"❌ '{feature}' is neither numerical nor categorical. Data type: {df[feature].dtype}")

    # -----------------------------
    # Numerical Analysis Helpers
    # -----------------------------
    @staticmethod
    def _analyze_numerical(df, feature, show_plots, bins, skew_thresh, kurt_thresh):
        DataExplorer._print_dtype_and_unique(df, feature)
        DataExplorer._print_summary_stats(df, feature)
        skew, kurt = DataExplorer._print_distribution_metrics(df, feature, skew_thresh, kurt_thresh)
        DataExplorer._detect_outliers(df, feature, skew, kurt, skew_thresh, show_plots)
        DataExplorer._print_missing_values(df, feature)

        if show_plots:
            DataExplorer._plot_numerical(df, feature, bins)

    @staticmethod
    def _print_dtype_and_unique(df, feature):
        dtype = df[feature].dtype
        unique_count = df[feature].dropna().nunique()
        print(f"📘 Data Type: {dtype}")
        print(f"💎 Unique Non-NA Values: {unique_count}")

    @staticmethod
    def _print_summary_stats(df, feature):
        stats = df[feature].describe()
        print("📊 Summary Statistics:")
        display(stats.to_frame().T.style.format("{:.2f}"))

    @staticmethod
    def _print_distribution_metrics(df, feature, skew_thresh, kurt_thresh):
        skew = df[feature].skew()
        kurt = df[feature].kurtosis()
        print(f"📈 Skewness: {skew:.2f} {'⚠️' if abs(skew) > skew_thresh else ''}")
        print(f"📈 Kurtosis: {kurt:.2f} {'⚠️' if kurt > kurt_thresh else ''}")
        return skew, kurt

    @staticmethod
    def _detect_outliers(df, feature, skew, kurt, skew_thresh, show_sample=False):
        if abs(skew) > skew_thresh or abs(kurt) > 2:
            method = "IQR Method"
            Q1, Q3 = df[feature].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = df[(df[feature] < Q1 - 1.5 * IQR) | (df[feature] > Q3 + 1.5 * IQR)]
        else:
            method = "Z-Score Method"
            z_scores = zscore(df[feature].dropna())
            outliers = df[abs(z_scores) > 3]

        print(f"\n🔍 Outlier Detection: {method}")
        if outliers.empty:
            print("   └── ✅ No outliers detected.")
        else:
            print(f"   └── ⚠️ {len(outliers)} outliers found ({len(outliers)/len(df)*100:.2f}% of rows)")
            if show_sample:
                display(outliers[[feature]].head())

    @staticmethod
    def _print_missing_values(df, feature):
        missing_count = df[feature].isnull().sum()
        if missing_count == 0:
            print("✅ No rows with missing values found.")
        else:
            total_rows = len(df)
            print(f"⚠️ Missing values: {missing_count}/{total_rows} ({missing_count/total_rows*100:.2f}%)")
            display(df[df[feature].isnull()].head())

    @staticmethod
    def _plot_numerical(df, feature, bins):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        sns.histplot(df[feature], bins=bins, kde=True, ax=axes[0, 0], color="#4C72B0")
        axes[0, 0].set_title("Histogram + KDE")
        sns.boxplot(y=df[feature], ax=axes[0, 1], color='#C44E52')
        axes[0, 1].set_title("Boxplot")
        sns.violinplot(y=df[feature], ax=axes[1, 0], color='#55A868')
        axes[1, 0].set_title("Violin Plot")
        sm.qqplot(df[feature].dropna(), line='s', ax=axes[1, 1], markerfacecolor='#FFA500', markeredgecolor='black')
        axes[1, 1].set_title("QQ Plot")
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Categorical Analysis Helpers
    # -----------------------------
    @staticmethod
    def _analyze_categorical(df, feature, show_plots, top_n_pie, high_card_threshold, rare_threshold):
        dtype = df[feature].dtype
        unique_values = df[feature].dropna().unique()
        print(f"📘 Data Type: {dtype}")
        print(f"💎 Unique Non-NA Values: {len(unique_values)}")
        print(f"📋 Unique Values List: {list(unique_values)}")

        counts = df[feature].value_counts()
        percentages = df[feature].value_counts(normalize=True) * 100
        freq_table = pd.DataFrame({
            'Count': counts.apply(lambda x: f"{x:,}"),
            'Percentage (%)': percentages.round(2)
        })
        freq_table.index.name = None
        print("📊 Frequency Table:")
        display(freq_table)

        DataExplorer._categorical_warnings(df, feature, counts, unique_values, high_card_threshold, rare_threshold)

        if show_plots:
            DataExplorer._plot_categorical(df, feature, counts, top_n_pie)

    @staticmethod
    def _categorical_warnings(df, feature, counts, unique_values, high_card_threshold, rare_threshold):
        if len(unique_values) == 1:
            print(f"⚠️ Constant Value: Feature is constant with value: {unique_values[0]}")
        elif len(unique_values) > high_card_threshold:
            print(f"⚠️ High Cardinality (>{high_card_threshold}): {len(unique_values)} unique categories")

        percentages = counts / counts.sum() * 100
        rare_cats = percentages[percentages < (rare_threshold * 100)]
        if not rare_cats.empty:
            print(f"⚠️ Rare Categories (<{rare_threshold*100:.0f}%): {len(rare_cats)} found")

        # Whitespace / capitalization issues
        stripped = df[feature].astype(str).str.strip()
        if not df[feature].astype(str).equals(stripped):
            print("⚠️ Detected leading/trailing whitespace.")
        lowercase = df[feature].astype(str).str.lower()
        if len(lowercase.unique()) < len(df[feature].astype(str).unique()):
            print("⚠️ Potential inconsistent capitalization.")

        DataExplorer._print_missing_values(df, feature)

    @staticmethod
    def _plot_categorical(df, feature, counts, top_n_pie):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
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
        plt.tight_layout()
        plt.show()
