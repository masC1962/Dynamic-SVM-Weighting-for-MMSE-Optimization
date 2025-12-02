# README

# 📘 SVM动态加权对MMSE量表的优化

本项目提供两套完整的 MMSE（简易精神状态检查）数据分析流程：

1. **基于机器学习的 ROC 曲线分析**（按文化程度分组）
2. **按文化程度进行自定义加权评分的分析**（使用 JSON 配置文件）

两种分析均会自动生成结构化的结果文件，包括 Excel 结果与图形输出。

---

## 📁 项目结构

```
project/
│── script/
│   ├── run.bat                # 运行机器学习分析（Windows）
│   └── run_by_level.sh        # 运行按学历加权分析（Linux / macOS）
│
│── src/
│   ├── run.py                 # 机器学习模型训练 + ROC 曲线绘制
│   ├── run_by_level.py        # 按学历加权的 MMSE 分析脚本
│   └── weights_by_level.json  # 不同学历的加权配置
│
│── data/
│   ├── 文盲.xlsx
│   ├── 小学.xlsx
│   ├── 中学.xlsx
│   ├── 大学.xlsx
│   └── 亳州市社区调研MMSE.xlsx
│
└── results/                    # 输出目录（执行后自动生成）
```

---

# 1️⃣ 机器学习分析（run.py）

该流程会对不同文化程度的数据分别训练多种机器学习模型，并生成 ROC 曲线。

---

## 🔍 功能说明

### ✔ 读取四个 Excel 文件（按文化程度分类）：

* 文盲.xlsx
* 小学.xlsx
* 中学.xlsx
* 大学.xlsx

### ✔ 训练五种机器学习模型：

* 逻辑回归（Logistic Regression）
* 支持向量机（SVM）
* 决策树
* 随机森林（Random Forest）
* 梯度提升树（GBDT）

### ✔ 输出结果：

* 每个模型 × 每个文化程度的 ROC 曲线
* 结果汇总 Excel 文件：`结果.xlsx`

---

## ▶️ 运行方式

### **Windows（推荐）**

直接双击：

```
script/run.bat
```

或手动运行：

```bash
python src/run.py --data-dir data --output-dir results
```

---

# 2️⃣ 按文化程度加权的 MMSE 分析（run_by_level.py）

该流程使用 **不同文化程度对应的自定义加权值与阈值** 来评估 MMSE 结果。

加权参数定义在：

```
src/weights_by_level.json
```

### 示例 JSON：

```json
"文盲": {
    "时间": 1,
    "空间": 3,
    "记忆": 2,
    ...
    "阈值": 30
}
```

---

## ✔ 程序功能

* 读取单个 Excel 文件（可包含多个 sheet）
* 自动标准化列名（兼容不同格式）
* 按学历应用不同的评分权重
* 计算：

  * 自定义加权分数
  * 自定义阈值/预设阈值
  * 加权评分准确率
  * 分数分布
* 计算传统 MMSE 总分的准确率
* 输出带多个 sheet 的结果 Excel：

```
MMSE分析结果_自定义加权.xlsx
```

包含：

1. 准确率对比
2. 数据汇总
3. 加权值配置
4. 自定义评分详细
5. 原始 MMSE 评分详细

---

## ▶️ 运行方式

### **Linux / macOS**

```bash
bash script/run_by_level.sh
```

### **手动运行**

```bash
python src/run_by_level.py \
    --data-path data/亳州市社区调研MMSE.xlsx \
    --output-dir results \
    --weights-file src/weights_by_level.json
```

---

# 📦 环境依赖

需要安装以下 Python 包：

```bash
pip install numpy pandas matplotlib scikit-learn plottable openpyxl
```

---

# 📊 输出示例

执行两套分析后，`results/` 文件夹中将包含：

```
结果.xlsx                          # 机器学习分析结果
MMSE分析结果_自定义加权.xlsx        # 加权评分分析结果
SVM小学ROC.png
随机森林大学ROC.png
...
```

---

# 📝 许可证

本项目可自由使用、修改和分发。