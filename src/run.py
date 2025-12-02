# 数据预处理
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt   
import pandas as pd 
from plottable import Table
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from numpy import interp
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

def ROC(df_X, df_y, model, name, sch):
    """绘制ROC曲线并计算AUC"""
    # 定义n折交叉验证
    KF = KFold(n_splits=5, shuffle=True, random_state=1962)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    
    # 数据预处理：将"是"/"否"转换为1/0
    date = {"是", "否"}
    replace_date = {"是": 1, "否": 0}
    df_y = df_y.apply(lambda x: replace_date[x] if x in date else x)
    
    # 存储每一折的评估结果
    fold_results = []
    
    for train_index, test_index in KF.split(df_X):
        # 划分训练集和测试集
        X_train, X_test = df_X.iloc[train_index], df_X.iloc[test_index]
        Y_train, Y_test = df_y.iloc[train_index], df_y.iloc[test_index]
        
        # 训练模型
        model.fit(X_train, Y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
        
        # 插值处理
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        # 计算AUC
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # 计算其他评估指标
        report = classification_report(Y_test, y_pred, output_dict=True)
        fold_results.append({
            'fold': i,
            'auc': roc_auc,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1-score': report['1']['f1-score']
        })
        
        # 绘制当前折的ROC曲线
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
        i += 1

    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    
    # 计算平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # 绘制平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, color='b', 
             label=r'Mean ROC (area=%0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
             lw=2, alpha=.8)
    
    # 绘制标准差区域
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    
    # 设置图形属性
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - {sch} (5-fold Cross Validation)')
    plt.legend(loc='lower right')
    
    return mean_auc, std_auc, fold_results


def grid_search_svm(X_train, y_train):
    """为SVM执行网格搜索找到最佳参数"""
    print("执行SVM网格搜索...")
    print("参数空间：")
    print("  - C (惩罚因子): [0.1, 0.5, 1, 5, 10]")
    print("  - gamma (核系数): [0.01, 0.05, 0.1, 0.5, 1]")
    print("  - 总参数组合: 5 × 5 = 25种")
    
    # 按照您提供的参数空间定义
    param_grid = {
        'C': [0.1, 0.5, 1, 5, 10],
        'gamma': [0.01, 0.05, 0.1, 0.5, 1],
        'kernel': ['rbf']  # 固定使用RBF核函数
    }
    
    # 创建SVM模型
    svm = SVC(probability=True, random_state=1962)
    
    # 创建网格搜索对象
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=5,  # 5折交叉验证
        scoring='accuracy',
        n_jobs=-1,  # 使用所有可用的CPU核心
        verbose=1,
        return_train_score=True  # 返回训练分数用于分析
    )
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 打印所有参数组合的结果
    print("\n所有参数组合的交叉验证结果:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    # 显示前10个最佳结果
    for i in range(min(10, len(results_df))):
        params = results_df.iloc[i]['params']
        mean_score = results_df.iloc[i]['mean_test_score']
        std_score = results_df.iloc[i]['std_test_score']
        rank = results_df.iloc[i]['rank_test_score']
        print(f"排名{rank}: C={params['C']}, gamma={params['gamma']}, "
              f"分数={mean_score:.4f} ± {std_score:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def grid_search_dt(X_train, y_train):
    """为决策树执行网格搜索"""
    print("执行决策树网格搜索...")
    
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    dt = DecisionTreeClassifier(random_state=1962)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    return grid_search.best_estimator_


def grid_search_rf(X_train, y_train):
    """为随机森林执行网格搜索"""
    print("执行随机森林网格搜索...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=1962)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    return grid_search.best_estimator_


def grid_search_gbdt(X_train, y_train):
    """为GBDT执行网格搜索"""
    print("执行GBDT网格搜索...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    gbdt = GradientBoostingClassifier(random_state=1962)
    grid_search = GridSearchCV(gbdt, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    return grid_search.best_estimator_


def main(data_dir, output_dir=None):
    """主函数，执行所有分析"""
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # 构建完整的文件路径
    def get_file_path(filename):
        return os.path.join(data_dir, filename)
    
    # 如果没有指定输出目录，使用数据目录
    if output_dir is None:
        output_dir = data_dir
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 读取数据文件
    print("正在读取数据文件...")
    df0 = pd.read_excel(get_file_path("文盲.xlsx"))
    df0.dropna(axis=0, how='any', subset=None, inplace=True)
    df0 = df0.reset_index(drop=True)
    
    df1 = pd.read_excel(get_file_path("小学.xlsx"))
    df1.dropna(axis=0, how='any', subset=None, inplace=True)
    df1 = df1.reset_index(drop=True)
    
    df2 = pd.read_excel(get_file_path("中学.xlsx"))
    df2.dropna(axis=0, how='any', subset=None, inplace=True)
    df2 = df2.reset_index(drop=True)
    
    df3 = pd.read_excel(get_file_path("大学.xlsx"))
    df3.dropna(axis=0, how='any', subset=None, inplace=True)
    df3 = df3.reset_index(drop=True)
    
    df = pd.concat([df0, df1, df2, df3])
    df = df.reset_index(drop=True)

    # 取出不同文化程度的数据
    df_sum = []
    df_sum.append(df0)
    df_sum.append(df1)
    df_sum.append(df2)
    df_sum.append(df3)

    # 计算mmse得到的准确率
    ind = [17, 20, 22, 23]
    acc_mmse = []
    for k in range(4):
        tmp = [0, 0, 0, 0]
        for i in range(len(df_sum[k])):
            if (df_sum[k]['是否评分为痴呆'][i] == '否' and df_sum[k]['是否诊断为痴呆'][i] == '否'):
                tmp[0] += 1
            if (df_sum[k]['是否评分为痴呆'][i] == '否' and df_sum[k]['是否诊断为痴呆'][i] == '是'):
                tmp[1] += 1
            if (df_sum[k]['是否评分为痴呆'][i] == '是' and df_sum[k]['是否诊断为痴呆'][i] == '否'):
                tmp[2] += 1
            if (df_sum[k]['是否评分为痴呆'][i] == '是' and df_sum[k]['是否诊断为痴呆'][i] == '是'):
                tmp[3] += 1
        acc = (tmp[0] + tmp[3]) / len(df_sum[k])
        acc_mmse.append('%.5f' % acc)

    col = ['文盲', '小学', '中学', '大学']
    
    # 准备特征和标签
    df_y = []
    df_X = []
    for ind in range(4):
        df_sum[ind].drop('学历', axis=1, inplace=True)
        df_sum[ind].drop('是否评分为痴呆', axis=1, inplace=True)
        df_sum[ind].drop('总分', axis=1, inplace=True)
        
        # 分离特征和标签
        df_tmp_y = df_sum[ind]['是否诊断为痴呆']
        df_y.append(df_tmp_y)
        df_tmp_X = df_sum[ind].drop('是否诊断为痴呆', axis=1)
        df_X.append(df_tmp_X)

    # 存储所有模型的结果
    all_results = {
        'MMSE评定': acc_mmse,
        '逻辑回归': [],
        'SVM': [],
        '决策树': [],
        '随机森林': [],
        'GBDT': []
    }
    
    # 存储最佳参数
    best_params = {}
    svm_optimization_results = {}  # 存储SVM优化过程的详细结果

    # 训练和评估每个模型
    for ind in range(4):
        print(f"\n{'='*60}")
        print(f"分析文化程度: {col[ind]}")
        print(f"数据集大小: {len(df_X[ind])} 个样本")
        print(f"特征数量: {df_X[ind].shape[1]}")
        print(f"{'='*60}")
        
        # 数据预处理：将标签转换为数值
        date = {"是", "否"}
        replace_date = {"是": 1, "否": 0}
        y_processed = df_y[ind].apply(lambda x: replace_date[x] if x in date else x)
        
        # 1. 逻辑回归模型
        print(f"\n1. 训练逻辑回归模型...")
        logreg = LogisticRegression(max_iter=1000, random_state=1962)
        mean_auc, std_auc, fold_results = ROC(df_X[ind], df_y[ind], logreg, '逻辑回归', col[ind])
        logreg.fit(df_X[ind], y_processed)
        scores = cross_val_score(logreg, df_X[ind], y_processed, cv=5)
        all_results['逻辑回归'].append('%.5f ± %.5f' % (scores.mean(), scores.std()))
        #plt.savefig(os.path.join(output_dir, f'逻辑回归_{col[ind]}_ROC.png'), dpi=600, bbox_inches='tight')
        plt.clf()
        
        # 2. SVM模型（使用网格搜索）
        print(f"\n2. SVM超参数优化过程...")
        print(f"超参数优化 - {col[ind]}文化程度")
        print(f"数据集大小: {len(df_X[ind])}个样本")
        print(f"使用分层5折交叉验证进行网格搜索")
        
        # 首先划分训练集用于网格搜索
        X_train, X_val, y_train, y_val = train_test_split(
            df_X[ind], y_processed, test_size=0.2, random_state=1962, stratify=y_processed
        )
        
        print(f"训练集大小: {len(X_train)}个样本")
        print(f"验证集大小: {len(X_val)}个样本")
        
        # 执行网格搜索
        svm_best, svm_params = grid_search_svm(X_train, y_train)
        best_params[f'SVM_{col[ind]}'] = svm_params
        
        # 保存网格搜索的详细结果
        svm_optimization_results[col[ind]] = {
            'best_params': svm_params,
            'best_score': svm_best.score(X_val, y_val),
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
        
        print(f"\n最优参数验证性能:")
        print(f"最佳C值: {svm_params['C']}")
        print(f"最佳gamma值: {svm_params['gamma']}")
        print(f"验证集准确率: {svm_best.score(X_val, y_val):.4f}")
        
        # 在整个数据集上训练最佳SVM模型
        print(f"\n使用最优参数在整个数据集上训练最终SVM模型...")
        svm_best.fit(df_X[ind], y_processed)
        mean_auc, std_auc, fold_results = ROC(df_X[ind], df_y[ind], svm_best, 'SVM', col[ind])
        scores = cross_val_score(svm_best, df_X[ind], y_processed, cv=5)
        all_results['SVM'].append('%.5f ± %.5f' % (scores.mean(), scores.std()))
        #plt.savefig(os.path.join(output_dir, f'SVM_{col[ind]}_ROC.png'), dpi=600, bbox_inches='tight')
        plt.clf()
        
        print(f"最终模型5折交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")
        
        # 3. 决策树模型（使用网格搜索）
        print(f"\n3. 训练决策树模型（使用网格搜索）...")
        dt_best = grid_search_dt(X_train, y_train)
        best_params[f'决策树_{col[ind]}'] = dt_best.get_params()
        dt_best.fit(df_X[ind], y_processed)
        mean_auc, std_auc, fold_results = ROC(df_X[ind], df_y[ind], dt_best, '决策树', col[ind])
        scores = cross_val_score(dt_best, df_X[ind], y_processed, cv=5)
        all_results['决策树'].append('%.5f ± %.5f' % (scores.mean(), scores.std()))
        #plt.savefig(os.path.join(output_dir, f'决策树_{col[ind]}_ROC.png'), dpi=600, bbox_inches='tight')
        plt.clf()
        
        # 4. 随机森林模型（使用网格搜索）
        print(f"\n4. 训练随机森林模型（使用网格搜索）...")
        rf_best = grid_search_rf(X_train, y_train)
        best_params[f'随机森林_{col[ind]}'] = rf_best.get_params()
        rf_best.fit(df_X[ind], y_processed)
        mean_auc, std_auc, fold_results = ROC(df_X[ind], df_y[ind], rf_best, '随机森林', col[ind])
        scores = cross_val_score(rf_best, df_X[ind], y_processed, cv=5)
        all_results['随机森林'].append('%.5f ± %.5f' % (scores.mean(), scores.std()))
        #plt.savefig(os.path.join(output_dir, f'随机森林_{col[ind]}_ROC.png'), dpi=600, bbox_inches='tight')
        plt.clf()
        
        # 5. GBDT模型（使用网格搜索）
        print(f"\n5. 训练GBDT模型（使用网格搜索）...")
        gbdt_best = grid_search_gbdt(X_train, y_train)
        best_params[f'GBDT_{col[ind]}'] = gbdt_best.get_params()
        gbdt_best.fit(df_X[ind], y_processed)
        mean_auc, std_auc, fold_results = ROC(df_X[ind], df_y[ind], gbdt_best, 'GBDT', col[ind])
        scores = cross_val_score(gbdt_best, df_X[ind], y_processed, cv=5)
        all_results['GBDT'].append('%.5f ± %.5f' % (scores.mean(), scores.std()))
        #plt.savefig(os.path.join(output_dir, f'GBDT_{col[ind]}_ROC.png'), dpi=600, bbox_inches='tight')
        plt.clf()

    # 创建结果表格
    print("\n" + "="*60)
    print("正在生成结果表格...")
    demo_df = pd.DataFrame(all_results, index=col)
    
    # 保存结果到输出目录
    output_path = os.path.join(output_dir, "模型结果_带标准差.xlsx")
    demo_df.to_excel(output_path)
    print(f"结果已保存到: {output_path}")
    
    # 保存最佳参数
    params_df = pd.DataFrame.from_dict(best_params, orient='index')
    params_path = os.path.join(output_dir, "最佳参数.xlsx")
    params_df.to_excel(params_path)
    print(f"最佳参数已保存到: {params_path}")
    
    # 保存SVM优化过程的详细结果
    svm_opt_df = pd.DataFrame.from_dict(svm_optimization_results, orient='index')
    svm_opt_path = os.path.join(output_dir, "SVM优化过程.xlsx")
    svm_opt_df.to_excel(svm_opt_path)
    print(f"SVM优化过程已保存到: {svm_opt_path}")
    
    # 创建详细的结果报告
    print("\n" + "="*60)
    print("SVM超参数优化总结:")
    print("="*60)
    print("超参数空间:")
    print("  - 惩罚因子C: [0.1, 0.5, 1, 5, 10]")
    print("  - 核系数gamma: [0.01, 0.05, 0.1, 0.5, 1]")
    print("  - 核函数: RBF")
    print("  - 总参数组合: 5 × 5 = 25种")
    print("\n优化方法:")
    print("  - 使用分层5折交叉验证")
    print("  - 评估指标: 准确率")
    print("  - 选择策略: 选择平均交叉验证准确率最高的参数组合")
    
    print("\n各文化程度最优参数:")
    for edu in col:
        if edu in svm_optimization_results:
            params = svm_optimization_results[edu]['best_params']
            score = svm_optimization_results[edu]['best_score']
            print(f"  {edu}: C={params['C']}, gamma={params['gamma']}, 验证准确率={score:.4f}")
    
    print("\n" + "="*60)
    print("所有模型性能对比:")
    print("="*60)
    for model in ['MMSE评定', '逻辑回归', 'SVM', '决策树', '随机森林', 'GBDT']:
        print(f"\n{model}:")
        for i, edu in enumerate(col):
            if model in all_results:
                print(f"  {edu}: {all_results[model][i]}")
    
    print("\n训练/验证分割信息:")
    print("  - 网格搜索: 80%训练集 + 20%验证集")
    print("  - 模型评估: 5折交叉验证")
    print("  - 最终模型: 在整个数据集上训练")
    print("  - 随机种子: 1962 (确保结果可复现)")
    print("  - 分层抽样: 保持类别分布平衡")
    
    print("\n所有分析已完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行机器学习模型分析（带网格搜索）')
    parser.add_argument('--data-dir', '-d', type=str, default='../data',
                       help='数据文件目录路径 (默认: ../data)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='输出结果目录路径 (默认: 与数据目录相同)')
    
    args = parser.parse_args()
    
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出目录: {args.output_dir}")
    
    main(args.data_dir, args.output_dir)