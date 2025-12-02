# README

# 📘 Dynamic SVM Weighting for MMSE Optimization

This project provides two complete analysis pipelines for Mini-Mental State Examination (MMSE) data:

1. **Machine-learning–based ROC analysis** across four education levels (Illiterate, Primary School, Middle School, College).
2. **Custom weighted scoring analysis** using a JSON configuration for different education levels.

Both pipelines output structured results, including Excel summary files and diagnostic plots.

---

## 📁 Project Structure

```
project/
│── script/
│   ├── run.bat                # Run machine learning analysis (Windows)
│   └── run_by_level.sh        # Run weighted scoring analysis (Linux / macOS)
│
│── src/
│   ├── run.py                 # ML model training + ROC curve generation
│   ├── run_by_level.py        # Education-level–specific weighted scoring analysis
│   └── weights_by_level.json  # Weight & threshold configuration
│
│── data/
│   ├── 文盲.xlsx
│   ├── 小学.xlsx
│   ├── 中学.xlsx
│   ├── 大学.xlsx
│   └── 亳州市社区调研MMSE.xlsx
│
└── results/                    # Output directory (generated automatically)
```

---

# 1️⃣ Machine Learning Analysis (run.py)

This pipeline trains multiple machine learning models for each education level and generates ROC curves.

---

## 🔍 Features

### ✔ Reads four separate Excel files:

* 文盲.xlsx
* 小学.xlsx
* 中学.xlsx
* 大学.xlsx

### ✔ Trains five ML models:

* Logistic Regression
* SVM
* Decision Tree
* Random Forest
* Gradient Boosting (GBDT)

### ✔ Generates:

* ROC curves for each model × education level
* An Excel summary table `结果.xlsx`

---

## ▶️ How to Run

### **Windows**

Double-click:

```
script/run.bat
```

or run manually:

```bash
python src/run.py --data-dir data --output-dir results
```

---

# 2️⃣ Weighted MMSE Scoring (run_by_level.py)

This pipeline evaluates MMSE using **custom weights and thresholds** for each education level.

The weights are defined in:

```
src/weights_by_level.json
```

### Example JSON snippet:

```json
"文盲": {
    "时间": 1,
    "空间": 3,
    "记忆": 2,
    ...
    "阈值": 30
}
```

### ✔ What the script does:

* Reads **a single MMSE Excel file**
* Normalizes column names automatically
* Applies **different weights** for each education level
* Computes:

  * Weighted score
  * Accuracy
  * Threshold used
  * Score distributions
* Computes MMSE original score accuracy
* Outputs a multi-sheet Excel file:

```
MMSE分析结果_自定义加权.xlsx
```

Sheets include:

1. Accuracy comparison
2. Data summary
3. Weight configuration
4. Weighted scoring details
5. MMSE scoring details

---

## ▶️ How to Run Weighted Analysis

### **Linux / macOS**

```
bash script/run_by_level.sh
```

### **Manual execution**

```bash
python src/run_by_level.py \
    --data-path data/亳州市社区调研MMSE.xlsx \
    --output-dir results \
    --weights-file src/weights_by_level.json
```

---

# 📦 Dependencies

Install required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn plottable openpyxl
```

---

# 📊 Output Examples

After running both pipelines, the `results/` folder will contain:

```
结果.xlsx                          # ML model summary
MMSE分析结果_自定义加权.xlsx        # Weighted analysis results
SVM小学ROC.png
随机森林大学ROC.png
...
```

---

# 📝 License


This project is free to use, modify, and redistribute.
