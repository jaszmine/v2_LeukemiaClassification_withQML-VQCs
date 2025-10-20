# VERSION 2: Leukemia Classification with Quantum Machine Learning (VQCs)

An updated quantum machine learning project for leukemia classification using Variational Quantum Circuits (VQCs). This project demonstrates how to leverage quantum computing for medical data classification with 16 gene features.

Kaggle dataset used: [Gene expression dataset (Golub et al.)](https://www.kaggle.com/datasets/crawford/gene-expression?select=data_set_ALL_AML_independent.csv)

## Key Features
- 16 carefully selected biomarker genes
- 22 patients with perfect class balance (50% AML, 50% ALL)
- 80/20 train-test split with stratification
- Comprehensive data quality assessment

Chosen Genes we're analyzing:
```bash
aml_genes = ['M55150_at', 'X95735_at', 'Y12670_at', 'L08246_at', 
             'M62762_at', 'M63138_at', 'M57710_at', 'M81695_s_at']
all_genes = ['U22376_cds2_s_at', 'X59417_at', 'U05259_rna1_at', 'M31211_s_at', 
             'M91432_at', 'Z69881_at', 'D38073_at', 'U35451_at']
```

## Project Structure
```tree
V2_LEUKEMIACLASSIFICATION_withQML-VQCs/
│
├── data/
│   ├── raw/
│   │   ├── actual.csv
│   │   ├── data_set_ALL_AML_train.csv
│   │   └── data_set_ALL_AML_independent.csv
│   └── processed/
│
├── src/
│   ├── final_patient_gene_table.csv
│   ├── preprocessing.py
│   ├── selected_patients_list.csv
│   ├── visualize_preprocessing.py
│   └── plots/
│       ├── patient_distribution.png
│       ├── gene_expression_heatmap.png
│       ├── call_quality_heatmap.png
│       ├── pca_analysis.png
│       ├── expression_by_cancer.png
│       ├── correlation_matrix.png
│       ├── patient_clustering.png
│       ├── call_quality_summary.png
│       └── summary_statistics.png
│
├── qml_env/
│   └── (virtual environment files)
│
├── README.md
└── requirements.txt
```


## Running

### 1. Clone Repo & Navigate to Project Directory
```bash
git clone https://github.com/jaszmine/v2_LeukemiaClassification_withQML-VQCs.git
cd v2_LeukemiaClassification_withQML-VQCs
```

### 2. Create & Activate Virtual Environment

On macOS or Linux: 
```bash
python3 -m venv qml_env
source qml_env/bin/activate
```

On Windows: 
```bash
python3 -m venv qml_env
qml_venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 4. Run the Scripts
Added a feature that asks for how many genes we're analyzing. For now, just enter 16 and yes when asked for if using the default. The default consists of the original 16 we talked about (after stratefied sampling and categorizing into the 5 main sueprgroups). This feature is mainly for when we run the other experiments later on and start increasing the number of genes/qubits.
```bash
python3 preprocessing.py
Train data loaded successfully.
Labels data loaded successfully.
Enter the number of genes to test: 16
Use default probe IDs? (yes/no): yes
```

```bash
python3 visualize_preprocessed.py
```