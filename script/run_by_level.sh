#!/bin/bash

echo "MMSE数据分析程序（按学历不同加权）"
echo "================================"

# 设置路径
DATA_FILE="data/亳州市社区调研MMSE.xlsx"
OUTPUT_DIR="results"
WEIGHTS_FILE="src/weights_by_level.json"

# 运行Python脚本
python src/run_by_level.py --data-path "$DATA_FILE" --output-dir "$OUTPUT_DIR" --weights "$WEIGHTS_FILE"

echo "分析完成！"
echo "结果保存在 $OUTPUT_DIR 文件夹中"