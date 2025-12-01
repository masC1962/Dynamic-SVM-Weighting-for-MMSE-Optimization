# README

# 📘 SVM动态加权对MMSE量表的优化

本项目用于对不同文化程度（文盲、小学、中学、大学）的人群进行机器学习模型训练，并生成 ROC 曲线与性能评估表格。
分析内容包含多种模型：Logistic Regression、SVM、决策树、随机森林、GBDT，并将结果自动保存到指定目录。

---

## 📁 项目结构

```
project/
│── script/
│   └── run.bat               # 运行主程序的批处理脚本
│
│── src/
│   └── run.py                # 主分析程序
│
│── data/                     # 输入数据（Excel 文件）
│   ├── 文盲.xlsx
│   ├── 小学.xlsx
│   ├── 中学.xlsx
│   └── 大学.xlsx
│
└── results/                  # 输出的结果（首次运行自动创建）
```

---

## 📜 运行脚本（script/run.bat）

```bat
@echo off
echo 正在运行数据分析程序...

:: 运行Python脚本，传入data目录路径
python src/run.py --data-dir data --output-dir results

echo 运行完成！
pause
```

此脚本会自动调用 `run.py` 并将分析结果输出到 `results/` 文件夹。

---

## 🔍 程序功能说明（src/run.py）

`run.py` 会执行以下任务：

### 1. **读取数据**

从 `data/` 目录读取 4 个 Excel 文件：

* 文盲.xlsx
* 小学.xlsx
* 中学.xlsx
* 大学.xlsx

并自动清洗数据（删除空值行、重置索引）。

---

### 2. **模型训练（5 折交叉验证）**

程序对每组文化程度分别训练下列模型：

* Logistic Regression
* SVM
* 决策树
* 随机森林
* GBDT

每个模型会计算：

* **准确率（交叉验证）**
* **ROC 曲线**
* **AUC 值**

---

### 3. **绘制 ROC 曲线**

程序会为每个模型生成：

```
{模型名}{文化程度}ROC.png
```

并自动移动到输出目录。

---

### 4. **生成最终结果 Excel**

输出文件：

```
结果.xlsx
```

内容包含：

| 文化程度 | MMSE评定 | Logistic 回归 | SVM | 决策树 | 随机森林 | GBDT |
| ---- | ------ | ----------- | --- | --- | ---- | ---- |
| 文盲   | ...    | ...         | ... | ... | ...  | ...  |
| 小学   | ...    | ...         | ... | ... | ...  | ...  |
| 中学   | ...    | ...         | ... | ... | ...  | ...  |
| 大学   | ...    | ...         | ... | ... | ...  | ...  |

---

## ▶️ 如何运行

### **方式一：直接双击运行脚本（推荐）**

双击：

```
script/run.bat
```

程序会：

* 自动读取 `data/` 下的数据
* 在 `results/` 下生成 ROC 图片与 Excel 结果

---

### **方式二：命令行手动运行**

```bash
python src/run.py --data-dir data --output-dir results
```

如果不指定输出路径：

```bash
python src/run.py --data-dir data
```

则所有输出会保存在 `data/` 下。

---

## 📦 环境依赖

请安装以下 Python 库：

```bash
pip install numpy pandas matplotlib scikit-learn plottable
```

⚠️ **注意：部分 ROC 绘图需要中文字体（SimHei）支持，否则可能出现乱码。**

---

## 📊 输出示例

程序运行后，`results/` 目录将包含：

```
结果.xlsx
逻辑回归文盲ROC.png
逻辑回归小学ROC.png
……
GBDT大学ROC.png
```

---

## 📝 许可证

本项目可自由修改和扩展。
