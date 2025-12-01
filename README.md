# README

# 📘 Dynamic SVM Weighting for MMSE Optimization

This project provides a complete machine-learning–based analysis workflow for evaluating cognitive assessment data across four education levels: **Illiterate**, **Primary School**, **Middle School**, and **College**.

The program trains multiple models, generates ROC curves, and outputs performance metrics into an Excel report.

---

## 📁 Project Structure

```
project/
│── script/
│   └── run.bat               # Batch script to execute the analysis
│
│── src/
│   └── run.py                # Main Python analysis script
│
│── data/                     # Input data (Excel files)
│   ├── 文盲.xlsx
│   ├── 小学.xlsx
│   ├── 中学.xlsx
│   └── 大学.xlsx
│
└── results/                  # Output directory
```

---

## 📜 Batch Script (script/run.bat)

```bat
@echo off
echo Running data analysis...

:: Execute Python script with data directory
python src/run.py --data-dir data --output-dir results

echo Done!
pause
```

This script automatically runs the analysis and stores all results in the `results/` folder.

---

## 🔍 What the Analysis Script Does (src/run.py)

The `run.py` program performs the full machine-learning workflow:

---

### **1. Data Loading**

Reads 4 Excel datasets:

* 文盲.xlsx (Illiterate)
* 小学.xlsx (Primary School)
* 中学.xlsx (Middle School)
* 大学.xlsx (College)

Data is cleaned by removing rows with missing values and resetting indices.

---

### **2. Model Training (5-Fold Cross-Validation)**

For each education level, the following models are trained:

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Gradient Boosting (GBDT)

For every model, the script computes:

* **Cross-validation accuracy**
* **ROC curve**
* **AUC value**

---

### **3. ROC Curve Generation**

For each model and education level, the script generates a figure:

```
{ModelName}{EducationLevel}ROC.png
```

All images are automatically moved to the output directory.

---

### **4. Final Excel Report**

A summary table is saved as:

```
results.xlsx
```

Columns include:

| Education Level | MMSE Accuracy | Logistic Regression | SVM | Decision Tree | Random Forest | GBDT |
| --------------- | ------------- | ------------------- | --- | ------------- | ------------- | ---- |

---

## ▶️ How to Run

### ✔️ **Method 1: Double-click the batch file (Recommended)**

Run:

```
script/run.bat
```

This will:

* Read data from `data/`
* Save all results to `results/`

---

### ✔️ **Method 2: Run manually from command line**

```bash
python src/run.py --data-dir data --output-dir results
```

If no output directory is specified:

```bash
python src/run.py --data-dir data
```

results will be saved in the data folder.

---

## 📦 Dependencies

Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn plottable
```

⚠️ For ROC plot labels, ensure your environment supports **Chinese font “SimHei”** to avoid garbled text.

---

## 📊 Output Files

After running the script, the `results/` folder will contain:

```
results.xlsx
LogisticRegression_Illiterate_ROC.png
LogisticRegression_PrimarySchool_ROC.png
...
GBDT_College_ROC.png
```

---

## 📝 License

This project is free to use, modify, and extend.
