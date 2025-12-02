# MMSE数据分析程序（按学历不同加权参数）
import pandas as pd 
import numpy as np
import os
import sys
import argparse
import json

def calculate_custom_score(row, weights):
    """计算自定义加权分数"""
    score = 0
    for col, weight in weights.items():
        if col in row.index:
            # 确保值是数值类型
            try:
                value = float(row[col])
                score += value * weight
            except (ValueError, TypeError):
                # 如果转换失败，跳过该项
                continue
    return score

def calculate_accuracy(df_level, weights):
    """计算自定义加权评分的准确率"""
    if len(df_level) == 0:
        return 0.0
    
    # 计算自定义分数
    df_level = df_level.copy()
    df_level['自定义分数'] = df_level.apply(lambda row: calculate_custom_score(row, weights), axis=1)
    
    # 使用JSON中定义的阈值进行判断
    threshold = weights.get('阈值', df_level['自定义分数'].median())
    
    # 预测结果：根据阈值判断是否痴呆
    df_level['预测结果'] = df_level['自定义分数'].apply(lambda x: '是' if x < threshold else '否')
    
    # 确保标签是字符串类型并去除空格
    df_level['是否诊断为痴呆'] = df_level['是否诊断为痴呆'].astype(str).str.strip()
    df_level['预测结果'] = df_level['预测结果'].astype(str).str.strip()
    
    # 计算准确率
    correct = len(df_level[df_level['是否诊断为痴呆'] == df_level['预测结果']])
    accuracy = correct / len(df_level)
    
    return accuracy

def load_weights_config(weights_file):
    """从JSON文件加载加权值和阈值配置"""
    weights_dict = {}
    
    try:
        with open(weights_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"从JSON文件加载配置: {weights_file}")
        
        if isinstance(config, dict):
            # 检查是否为嵌套结构（按学历分别配置）
            if '文盲' in config or '小学' in config:
                # 格式1: {"文盲": {"时间": 1.0, "阈值": 20.5, ...}, "小学": {...}, ...}
                for level in ['文盲', '小学', '中学', '大学']:
                    if level in config:
                        weights_dict[level] = config[level]
                        print(f"  找到 {level} 的配置")
                    else:
                        # 如果没有该学历的配置，使用默认值
                        weights_dict[level] = {}
                        print(f"  警告: {level} 没有配置，使用空配置")
            else:
                # 格式2: {"时间": 1.0, "空间": 1.2, "阈值": 22.0, ...} - 所有学历使用相同配置
                for level in ['文盲', '小学', '中学', '大学']:
                    weights_dict[level] = config.copy()
                print("  所有学历使用相同配置")
        else:
            print(f"错误: JSON文件格式不正确")
            sys.exit(1)
            
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        sys.exit(1)
    
    # 确保所有学历都有完整的配置，缺失的使用默认值
    default_keys = ['时间', '空间', '记忆', '计算', '延迟回忆', '命名', '复述', '执行', '阅读', '书写', '空间结构']
    
    for level in ['文盲', '小学', '中学', '大学']:
        if level not in weights_dict:
            weights_dict[level] = {}
        
        level_weights = weights_dict[level]
        
        # 确保所有测试项目都有加权值
        for key in default_keys:
            if key not in level_weights:
                level_weights[key] = 1.0
                print(f"  警告: {level} 的 {key} 使用默认值 1.0")
        
        # 确保有阈值配置
        if '阈值' not in level_weights:
            # 如果没有阈值，标记为需要计算
            level_weights['阈值'] = None
            print(f"  注意: {level} 的阈值将使用数据中位数计算")
    
    return weights_dict

def main(data_path, output_dir, weights_file, sheet_name=None):
    """主函数"""
    
    print(f"数据文件: {data_path}")
    print(f"输出目录: {output_dir}")
    print(f"加权配置文件: {weights_file}")
    
    # 检查文件是否存在
    if not os.path.exists(weights_file):
        print(f"错误: 加权配置文件不存在: {weights_file}")
        sys.exit(1)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 读取Excel文件
    print("正在读取数据文件...")
    try:
        if sheet_name:
            df_all = pd.read_excel(data_path, sheet_name=sheet_name)
        else:
            # 尝试读取所有sheet，合并数据
            xls = pd.ExcelFile(data_path)
            sheet_names = xls.sheet_names
            
            df_list = []
            for sheet in sheet_names:
                df_sheet = pd.read_excel(xls, sheet_name=sheet)
                df_list.append(df_sheet)
            
            df_all = pd.concat(df_list, ignore_index=True)
            print(f"合并了 {len(sheet_names)} 个sheet的数据")
            
    except Exception as e:
        print(f"读取文件失败: {e}")
        sys.exit(1)
    
    # 重命名列名，使其标准化
    print("正在标准化列名...")
    column_mapping = {}
    for col in df_all.columns:
        col_str = str(col).strip()
        if any(keyword in col_str for keyword in ['时间', '1时间']):
            column_mapping[col] = '时间'
        elif any(keyword in col_str for keyword in ['空间', '2空间']):
            column_mapping[col] = '空间'
        elif any(keyword in col_str for keyword in ['记忆', '3记忆']):
            column_mapping[col] = '记忆'
        elif any(keyword in col_str for keyword in ['计算', '4计算']):
            column_mapping[col] = '计算'
        elif any(keyword in col_str for keyword in ['延迟回忆', '5延迟回忆', '延迟', '回忆']):
            column_mapping[col] = '延迟回忆'
        elif any(keyword in col_str for keyword in ['命名', '6命名']):
            column_mapping[col] = '命名'
        elif any(keyword in col_str for keyword in ['复述', '7复述']):
            column_mapping[col] = '复述'
        elif any(keyword in col_str for keyword in ['执行', '8执行']):
            column_mapping[col] = '执行'
        elif any(keyword in col_str for keyword in ['阅读', '9阅读']):
            column_mapping[col] = '阅读'
        elif any(keyword in col_str for keyword in ['书写', '10书写']):
            column_mapping[col] = '书写'
        elif any(keyword in col_str for keyword in ['空间结构', '11空间结构']):
            column_mapping[col] = '空间结构'
        elif any(keyword in col_str for keyword in ['学历', '文化程度', 'A']):
            column_mapping[col] = '学历'
        elif any(keyword in col_str for keyword in ['临床诊断', '是否诊断为痴呆', 'N']):
            column_mapping[col] = '是否诊断为痴呆'
        elif any(keyword in col_str for keyword in ['分数诊断', '是否评分为痴呆', 'O']):
            column_mapping[col] = '是否评分为痴呆'
        elif '总分' in col_str or 'M' in col_str:
            column_mapping[col] = '总分'
    
    # 如果第一列看起来是学历数据而不是列名，特殊处理
    if len(df_all.columns) > 0:
        first_col = df_all.columns[0]
        first_col_str = str(first_col).strip()
        
        # 检查第一列的值是否包含学历信息
        if len(df_all) > 0:
            first_value = str(df_all.iloc[0, 0])
            if first_value in ['文盲', '小学', '中学', '大学'] and first_col_str not in column_mapping:
                print(f"检测到第一列可能是学历数据，值: {first_value}")
                # 将第一列重命名为'学历'，并将值作为数据
                column_mapping[first_col] = '学历'
    
    df_all = df_all.rename(columns=column_mapping)
    
    # 删除所有含空缺值的行
    df_all = df_all.dropna()
    df_all = df_all.reset_index(drop=True)
    
    print("数据列名:", df_all.columns.tolist())
    print(f"总数据量: {len(df_all)} 条")
    
    # 检查必要的列是否存在
    required_columns = ['学历', '是否诊断为痴呆', '是否评分为痴呆']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    
    if missing_columns:
        print(f"错误: 缺少必要列: {missing_columns}")
        print("尝试检查原始数据格式...")
        print("原始列名:", df_all.columns.tolist())
        print("前5行数据:")
        print(df_all.head())
        sys.exit(1)
    
    # 加载加权值配置
    print("\n" + "="*60)
    print("加权值配置:")
    print("="*60)
    
    weights_dict = load_weights_config(weights_file)
    
    # 显示每个学历的加权值配置
    for level in ['文盲', '小学', '中学', '大学']:
        print(f"\n{level} 的配置:")
        weights = weights_dict[level]
        
        # 显示测试项目加权值
        test_items = ['时间', '空间', '记忆', '计算', '延迟回忆', '命名', '复述', '执行', '阅读', '书写', '空间结构']
        print("  测试项目加权值:")
        for item in test_items:
            if item in weights:
                print(f"    {item}: {weights[item]}")
        
        # 显示阈值
        threshold = weights.get('阈值')
        if threshold is not None:
            print(f"  阈值: {threshold}")
        else:
            print(f"  阈值: 使用数据中位数")
    
    # 分组数据
    education_levels = ['文盲', '小学', '中学', '大学']
    df_by_level = {}
    
    for level in education_levels:
        df_level = df_all[df_all['学历'] == level].copy()
        if len(df_level) > 0:
            df_by_level[level] = df_level
            print(f"\n{level}: {len(df_level)} 条数据")
            print(f"  痴呆人数: {len(df_level[df_level['是否诊断为痴呆'] == '是'])}")
            print(f"  非痴呆人数: {len(df_level[df_level['是否诊断为痴呆'] == '否'])}")
        else:
            print(f"\n{level}: 0 条数据")
    
    # 计算自定义加权分数准确率（按学历使用不同加权值和阈值）
    print("\n" + "="*60)
    print("计算自定义加权分数准确率:")
    print("="*60)
    
    custom_accuracies = []
    custom_thresholds = []
    custom_details = []  # 保存详细结果
    
    for level in education_levels:
        if level in df_by_level:
            df_level = df_by_level[level]
            weights = weights_dict[level]
            
            accuracy = calculate_accuracy(df_level, weights)
            custom_accuracies.append('%.5f' % accuracy)
            
            # 获取实际使用的阈值
            threshold = weights.get('阈值')
            if threshold is None:
                # 计算中位数
                df_level_temp = df_level.copy()
                df_level_temp['自定义分数'] = df_level_temp.apply(lambda row: calculate_custom_score(row, weights), axis=1)
                threshold = df_level_temp['自定义分数'].median()
                custom_thresholds.append('%.2f' % threshold)
            else:
                custom_thresholds.append('%.2f' % threshold)
            
            print(f"{level}:")
            print(f"  准确率 = {accuracy:.5f}")
            print(f"  阈值 = {custom_thresholds[-1]}")
            
            # 保存详细结果
            df_level_temp = df_level.copy()
            df_level_temp['自定义分数'] = df_level_temp.apply(lambda row: calculate_custom_score(row, weights), axis=1)
            
            details = {
                '学历': level,
                '样本数': len(df_level),
                '准确率': accuracy,
                '阈值': float(custom_thresholds[-1]),
                '自定义分数范围': f"{df_level_temp['自定义分数'].min():.2f} - {df_level_temp['自定义分数'].max():.2f}",
                '自定义分数均值': f"{df_level_temp['自定义分数'].mean():.2f}",
                '自定义分数中位数': f"{df_level_temp['自定义分数'].median():.2f}"
            }
            custom_details.append(details)
            
        else:
            custom_accuracies.append('0.00000')
            custom_thresholds.append('N/A')
            print(f"{level}: 无数据")
    
    # 计算MMSE原始分数准确率
    print("\n" + "="*60)
    print("计算MMSE原始分数准确率:")
    print("="*60)
    
    mmse_accuracies = []
    mmse_thresholds = []
    mmse_details = []  # 保存详细结果
    
    for level in education_levels:
        if level in df_by_level:
            df_level = df_by_level[level]
            
            # 计算MMSE总分
            if '总分' in df_level.columns:
                # 使用已有的总分
                total_scores = df_level['总分']
            else:
                # 计算总分（各项目分数之和）
                score_columns = ['时间', '空间', '记忆', '计算', '延迟回忆', '命名', '复述', '执行', '阅读', '书写', '空间结构']
                available_columns = [col for col in score_columns if col in df_level.columns]
                if available_columns:
                    total_scores = df_level[available_columns].sum(axis=1)
                else:
                    print(f"警告: {level} 没有找到任何测试项目列")
                    mmse_accuracies.append('0.00000')
                    mmse_thresholds.append('N/A')
                    continue
            
            # 确定阈值
            preset_thresholds = [17, 20, 22, 23]
            threshold = preset_thresholds[education_levels.index(level)]
            mmse_thresholds.append('%.2f' % threshold)
            
            # 预测结果：分数低于阈值预测为痴呆
            predictions = total_scores.apply(lambda x: '是' if x < threshold else '否')
            
            # 计算准确率
            correct = sum((df_level['是否诊断为痴呆'].astype(str).str.strip() == predictions.astype(str).str.strip()))
            accuracy = correct / len(df_level)
            mmse_accuracies.append('%.5f' % accuracy)
            
            print(f"{level}:")
            print(f"  准确率 = {accuracy:.5f}")
            print(f"  阈值 = {threshold:.2f}")
            
            # 保存详细结果
            details = {
                '学历': level,
                '样本数': len(df_level),
                '准确率': accuracy,
                '阈值': float(threshold),
                '总分范围': f"{total_scores.min():.2f} - {total_scores.max():.2f}",
                '总分均值': f"{total_scores.mean():.2f}",
                '总分中位数': f"{total_scores.median():.2f}"
            }
            mmse_details.append(details)
            
        else:
            mmse_accuracies.append('0.00000')
            mmse_thresholds.append('N/A')
            print(f"{level}: 无数据")
    
    # 保存结果
    print("\n" + "="*60)
    print("生成结果表格:")
    print("="*60)
    
    # 1. 准确率对比表
    result_df = pd.DataFrame({
        '文化程度': education_levels,
        '自定义加权评分_准确率': custom_accuracies,
        '自定义加权评分_阈值': custom_thresholds,
        'MMSE原始评分_准确率': mmse_accuracies,
        'MMSE原始评分_阈值': mmse_thresholds
    })
    
    # 2. 加权值配置表（按学历）
    weights_data = []
    for level in education_levels:
        level_weights = weights_dict[level]
        for item, weight in level_weights.items():
            # 跳过非测试项目的键（如'阈值'）
            if item in ['时间', '空间', '记忆', '计算', '延迟回忆', '命名', '复述', '执行', '阅读', '书写', '空间结构', '阈值']:
                weights_data.append({
                    '文化程度': level,
                    '配置项': item,
                    '值': weight
                })
    weights_df = pd.DataFrame(weights_data)
    
    # 3. 数据汇总表
    data_summary = []
    for level in education_levels:
        if level in df_by_level:
            df_level = df_by_level[level]
            data_summary.append({
                '文化程度': level,
                '样本数': len(df_level),
                '痴呆人数': len(df_level[df_level['是否诊断为痴呆'] == '是']),
                '非痴呆人数': len(df_level[df_level['是否诊断为痴呆'] == '否']),
                '痴呆比例': f'{len(df_level[df_level["是否诊断为痴呆"] == "是"])/len(df_level)*100:.2f}%'
            })
        else:
            data_summary.append({
                '文化程度': level,
                '样本数': 0,
                '痴呆人数': 0,
                '非痴呆人数': 0,
                '痴呆比例': '0.00%'
            })
    data_summary_df = pd.DataFrame(data_summary)
    
    # 4. 详细结果表
    if custom_details:
        custom_details_df = pd.DataFrame(custom_details)
    else:
        custom_details_df = pd.DataFrame()
    
    if mmse_details:
        mmse_details_df = pd.DataFrame(mmse_details)
    else:
        mmse_details_df = pd.DataFrame()
    
    # 保存结果到Excel
    output_path = os.path.join(output_dir, "MMSE分析结果_自定义加权.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='准确率对比', index=False)
        data_summary_df.to_excel(writer, sheet_name='数据汇总', index=False)
        weights_df.to_excel(writer, sheet_name='加权值配置', index=False)
        
        if not custom_details_df.empty:
            custom_details_df.to_excel(writer, sheet_name='自定义评分详情', index=False)
        
        if not mmse_details_df.empty:
            mmse_details_df.to_excel(writer, sheet_name='MMSE评分详情', index=False)
    
    print(f"\n结果已保存到: {output_path}")
    print(f"包含以下sheet:")
    print(f"  1. 准确率对比 - 两种方法的准确率对比")
    print(f"  2. 数据汇总 - 各文化程度数据分布")
    print(f"  3. 加权值配置 - 从JSON文件加载的配置")
    print(f"  4. 自定义评分详情 - 自定义加权评分的详细结果")
    print(f"  5. MMSE评分详情 - MMSE原始评分的详细结果")
    
    # 显示主要结果
    print("\n" + "="*60)
    print("主要结果对比:")
    print("="*60)
    print(result_df.to_string(index=False))
    
    print("\n分析完成！")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MMSE数据分析程序（自定义加权评分）')
    parser.add_argument('--data-path', '-d', type=str, required=True,
                       help='Excel数据文件路径')
    parser.add_argument('--output-dir', '-o', type=str, default='./results',
                       help='输出目录路径 (默认: ./results)')
    parser.add_argument('--weights-file', '-w', type=str, required=True,
                       help='加权值JSON配置文件路径')
    parser.add_argument('--sheet-name', '-s', type=str, default=None,
                       help='Excel sheet名称 (默认: 合并所有sheet)')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        sys.exit(1)
    
    # 运行主程序
    main(args.data_path, args.output_dir, args.weights_file, args.sheet_name)