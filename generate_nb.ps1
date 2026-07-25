import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("# Eksperimen Dataset: SP 500 Analyst Rating"),
    new_markdown_cell("## 1. Import Library"),
    new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import LabelEncoder"),
    new_markdown_cell("## 2. Load Dataset"),
    new_code_cell("df = pd.read_csv('../namadataset_raw/sp_500_analyst_rating_and_price_target_accuracy.csv')\ndf.head()"),
    new_markdown_cell("## 3. Data Exploration & Cleaning"),
    new_code_cell("df.info()"),
    new_code_cell("# Memilih fitur yang relevan\nfeatures = ['from_grade', 'to_grade', 'action', 'implied_direction']\ntarget = 'accuracy_30d'\n\ndf = df.dropna(subset=[target] + features)"),
    new_markdown_cell("## 4. Preprocessing"),
    new_code_cell("le = LabelEncoder()\nfor col in ['from_grade', 'to_grade', 'action']:\n    df[col] = le.fit_transform(df[col].astype(str))"),
    new_code_cell("df.head()"),
    new_markdown_cell("## 5. Save Preprocessed Data"),
    new_code_cell("import os\nos.makedirs('namadataset_preprocessing', exist_ok=True)\ndf.to_csv('namadataset_preprocessing/sp_500_analyst_rating_and_price_target_accuracy_preprocessed.csv', index=False)\nprint('Data preprocessed saved!')")
])

with open('generate_nb.py', 'w') as f:
    f.write('import nbformat\nnb = ' + repr(nb.dict()) + '\nnbformat.write(nbformat.from_dict(nb), \"Eksperimen_Franscelino-Melvyn.ipynb\")')
