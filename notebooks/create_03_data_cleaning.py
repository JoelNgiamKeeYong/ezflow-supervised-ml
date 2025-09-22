import yaml
import nbformat as nbf
import pandas as pd
from pathlib import Path

# Get config
with open("../config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load your dataset
df = pd.read_csv("../" + config["data_file_path"])

# Specify target column
target_column = "target"  # replace with your actual target
feature_columns = [col for col in df.columns if col != target_column]

# Notebook file path
notebook_path = Path("03_data_cleaning_template.ipynb")

# Create a new notebook
nb = nbf.v4.new_notebook()

##########################################################################################################################################
##########################################################################################################################################
# 🧼🫧🧹 Add initial markdown for title, banner, and intro
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """
# **🧼🫧🧹 Data Cleaning**

<img src="../assets/banner_data_cleaning.png" style="width:95%">

- **Data Cleaning** is a critical step in preparing your dataset for meaningful analysis and modeling.  

- A well-cleaned dataset reduces bias, improves model performance, and prevents misleading conclusions.  

- This notebook uses **manual coding** to ensure each feature is properly reviewed and cleaned.  

- Make sure to update the `data_file_path`, `identifier_column` and `target_column` entries in your `config.yaml` file before running the notebook.
        """
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 📦 Import General Libraries
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
---
**📦 Import General Libraries**"""
    )
)
nb.cells.append(nbf.v4.new_code_cell("import pandas as pd"))

##########################################################################################################################################
##########################################################################################################################################
# ⚙️ Configure Imports
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
**⚙️ Configure Imports**"""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path().resolve().parent  # if running from folder with parent directory as project root
sys.path.append(str(project_root))"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 🔧 Configure Notebook
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
**🔧 Configure Notebook**"""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'  # Show all outputs in a cell"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 🔧 Import Pipeline Classes
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
**🔧 Import Pipeline Classes**"""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """from src.data_explorer import DataExplorer
from src.data_cleaner import DataCleaner

explorer = DataExplorer()
cleaner = DataCleaner()"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 🚀 Load Config from `config.yaml`
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
**🚀 Load Config from `config.yaml`**"""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """import yaml

config_path = "../config.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATA_FILE_PATH = config["data_file_path"]
IDENTIFIER_COLUMN = config["identifier_column"]
TARGET_COLUMN = config["target_column"]
RANDOM_STATE = config["random_state"]
TEST_SIZE = config["test_size"]"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 📥 Load Data into Pandas DataFrame
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
**📥 Load Data into Pandas DataFrame**"""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """RELATIVE_FILE_PATH = Path("../", DATA_FILE_PATH)

df = pd.read_csv(RELATIVE_FILE_PATH)
df.head()"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 🧼 1. Apply Snake Case
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
---
# **🧼 1. Apply Snake Case**

- Using `snake_case` aligns with PEP 8, Python’s official style guide for code readability and consistency.

- It ensures column names are consistent and easy to read, avoiding spaces or special characters that can break code.

- Standardized naming makes data manipulation, merging, and referencing columns in scripts much smoother.

- Following PEP 8 conventions improves code maintainability, readability, and reduces potential errors in data pipelines."""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """df_cleaned = cleaner.clean_all(df=df, mode="snake_case_columns")
df_cleaned.columns"""
    )
)
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """**└─ 💡 Observations / Insights ──**

- Column names successfully amended"""
    )
)


##########################################################################################################################################
##########################################################################################################################################
# 🧼 2. Rearrange Columns
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
---
# **🧼 2. Rearrange Columns**

- Move the **target column** to the front for easier reference.  

- Keeps the dataset organized and makes analysis or model training steps more intuitive."""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """# df_cleaned = cleaner.clean_all(df=df_cleaned, mode="rearrange_columns")
# df_cleaned.head(1)"""
    )
)
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """**└─ 💡 Observations / Insights ──**

- Columns successfully rearranged"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 🧼 3. Drop Irrelevant Features
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
---
# **🧼 3. Drop Irrelevant Features**

- Remove columns that do not contribute to the analysis (e.g., IDs).  

- Drop **post-hoc features** that won’t be available at prediction time to avoid **data leakage**.  

- Example: When predicting `resale_price`, exclude `resale_price_USD` since it’s just a transformed version of the target."""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        "df_cleaned.columns"
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """df_cleaned = cleaner.clean_all(df=df_cleaned, mode="irrelevant")
print("\\n", df_cleaned.columns)"""
    )
)
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """**└─ 💡 Observations / Insights ──**

- Irrelevant features successfully dropped"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 🧼 4. Explore target variable
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
---
# **🧼 4. Explore Target Variable**

- Examine the **distribution** of the target variable.  
  - For regression: check spread, skewness, and outliers.  
  - For classification: review class balance and frequency counts.  

- Detect potential **issues** (e.g., extreme outliers in regression, class imbalance in classification).  

- Consider whether **transformations** (e.g., log-scaling for regression, class grouping for classification) are needed to improve analysis or model performance."""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """# Get snake case version of target name
target_column_snake_case = cleaner.convert_column_names_to_snake_case(df=df[[TARGET_COLUMN]])
target_column_name = target_column_snake_case.columns[0]

explorer.perform_univariate_analysis(df=df_cleaned, feature=target_column_name, show_plots=True)"""
    )
)
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """**└─ 💡 Observations / Insights ──**

- `Observation 1`

- `Observation 2`"""
    )
)
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """---
## **└─ 🫧 Clean Feature (1)**

- `Cleaning Step 1`

- `Cleaning Step 2`"""
    )
)
nb.cells.append(
    nbf.v4.new_code_cell(
        """# df_cleaned = cleaner.clean_all(df=df_cleaned, mode="clean_target")
# df_cleaned"""
    )
)
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """**└─ 💡 Observations / Insights ──**

- `Observation 1`

- `Observation 2`"""
    )
)

##########################################################################################################################################
##########################################################################################################################################
# 📝 5. Function to generate Explore section for a column
def create_explore_section(col_name, section_number=4, features_dict=None):
    """
    Create a notebook section to explore a column, with numbered header,
    including a “Clean Feature” subsection.
    
    Parameters
    ----------
    col_name : str
        Column name to explore.
    section_number : int
        Number to use in the section header.
    features_dict : dict, optional
        Dictionary mapping column names to descriptions.
        
    Returns
    -------
    list
        A list of nbformat cells:
        [markdown_intro, code_cell, markdown_observations, markdown_clean_intro, code_clean_cell, markdown_clean_obs]
    """
    
    # --- Get description from dictionary ---
    if features_dict and col_name in features_dict:
        col_description = features_dict[col_name]
    else:
        col_description = "No description available (please update data dictionary)."
    
    # --- Explore section ---
    md_intro = (
        "---\n"
        "---\n"
        f"# **🧼 {section_number}. Explore `{col_name}`**\n\n"
        f"- {col_description}"
    )
    
    code_cell = f"explorer.perform_univariate_analysis(df=df_cleaned, feature=\"{col_name}\", show_plots=True)"
    
    md_obs = (
        "**└─ 💡 Observations / Insights ──**\n\n"
        "- `Observation 1`\n"
        "- `Observation 2`"
    )
    
    # --- Clean Feature subsection ---
    md_clean_intro = (
        "---\n"
        "## **└─ 🫧 Clean Feature (1)**\n\n"
        "- `Cleaning Step 1`\n"
        "- `Cleaning Step 2`"
    )
    
    code_clean_cell = (
        f"# df_cleaned = cleaner.clean_all(df=df_cleaned, mode=\"clean_{col_name}\")\n"
        "# df_cleaned"
    )
    
    md_clean_obs = (
        "**└─ 💡 Observations / Insights ──**\n\n"
        "- `Observation 1`\n\n"
        "- `Observation 2`"
    )
    
    return [
        nbf.v4.new_markdown_cell(md_intro),
        nbf.v4.new_code_cell(code_cell),
        nbf.v4.new_markdown_cell(md_obs),
        nbf.v4.new_markdown_cell(md_clean_intro),
        nbf.v4.new_code_cell(code_clean_cell),
        nbf.v4.new_markdown_cell(md_clean_obs)
    ]

##########################################################################################################################################
##########################################################################################################################################
# Add feature columns starting from section 5
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # adjust if needed
sys.path.append(str(project_root))

from src.data_cleaner import DataCleaner
cleaner = DataCleaner()

# Convert target to snake_case
target_column_snake = cleaner.convert_column_names_to_snake_case(
    df=pd.DataFrame(columns=[config["target_column"]])
).columns[0]

# Convert features to snake_case, excluding target
feature_columns_snake_case = cleaner.convert_column_names_to_snake_case(
    df=df[[col for col in feature_columns if col != config["target_column"]]]
).columns.tolist()

# Original dictionary from YAML
features_dict_raw = config["features"]

# Convert dictionary keys to snake_case
features_dict = {}
for k, v in features_dict_raw.items():
    k_snake = cleaner.convert_column_names_to_snake_case(
        df=pd.DataFrame(columns=[k])
    ).columns[0]
    features_dict[k_snake] = v

# Add feature columns starting from section 5
section_number = 5
for col in feature_columns_snake_case:
    if col == target_column_snake:  # double-check exclusion
        continue
    for cell in create_explore_section(col, section_number=section_number, features_dict=features_dict):
        nb.cells.append(cell)
    section_number += 1

##########################################################################################################################################
##########################################################################################################################################
# Save notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Notebook created: {notebook_path}")
