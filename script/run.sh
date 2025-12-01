@echo off
echo 正在运行数据分析程序...

:: 运行Python脚本，传入data目录路径
python src/run.py --data-dir data --output-dir results

echo 运行完成！
pause