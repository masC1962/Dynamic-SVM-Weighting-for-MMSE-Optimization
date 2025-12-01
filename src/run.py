# 数据预处理
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt   
import pandas as pd 
from plottable import Table
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import os
import sys
import argparse

def ROC(df_X, df_y, model, name, sch):
    # 定义n折交叉验证
    KF = KFold(n_splits=5, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    # data为数据集,利用KF.split划分训练集和测试集

    for train_index, test_index in KF.split(df_X):
        date = {"是", "否"}
        replace_date = {"是": 1, "否": 0}
        df_y = df_y.apply(lambda x: replace_date[x] if x in date else x)  # 合并文化程度
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = df_X.iloc[train_index], df_X.iloc[test_index]
        Y_train, Y_test = df_y.iloc[train_index], df_y.iloc[test_index]
        # 建立模型(模型已经定义)
        # model = LogisticRegression()
        # 训练模型
        model.fit(X_train, Y_train)
        # 利用model.predict获取测试集的预测值
        y_pred = model.predict(X_test)
        # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
        # interp:插值 把结果添加到tprs列表中
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
        i += 1

    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name + sch + 'ROC')
    plt.legend(loc='lower right')
    plt.savefig(name + sch + 'ROC.png', dpi=600, bbox_inches='tight')
    plt.clf()


def main(data_dir, output_dir=None):
    """主函数，执行所有分析"""
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
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
    df0 = pd.read_excel(get_file_path("文盲.xlsx"))  # 导入数据
    df0.dropna(axis=0, how='any', subset=None, inplace=True)  # 删除所有含空缺值的行
    df0 = df0.reset_index(drop=True)  # 更新索引
    
    df1 = pd.read_excel(get_file_path("小学.xlsx"))  # 导入数据
    df1.dropna(axis=0, how='any', subset=None, inplace=True)  # 删除所有含空缺值的行
    df1 = df1.reset_index(drop=True)  # 更新索引
    
    df2 = pd.read_excel(get_file_path("中学.xlsx"))  # 导入数据
    df2.dropna(axis=0, how='any', subset=None, inplace=True)  # 删除所有含空缺值的行
    df2 = df2.reset_index(drop=True)  # 更新索引
    
    df3 = pd.read_excel(get_file_path("大学.xlsx"))  # 导入数据
    df3.dropna(axis=0, how='any', subset=None, inplace=True)  # 删除所有含空缺值的行
    df3 = df3.reset_index(drop=True)  # 更新索引
    
    df = pd.concat([df0, df1, df2, df3])
    df = df.reset_index(drop=True)  # 更新索引

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
            if (df_sum[k]['是否评分为痴呆'][i] == '否' and df_sum[k]['是否诊断为痴呆'][i] == '否'):  # TN
                tmp[0] += 1
            if (df_sum[k]['是否评分为痴呆'][i] == '否' and df_sum[k]['是否诊断为痴呆'][i] == '是'):  # FN
                tmp[1] += 1
            if (df_sum[k]['是否评分为痴呆'][i] == '是' and df_sum[k]['是否诊断为痴呆'][i] == '否'):  # FP
                tmp[2] += 1
            if (df_sum[k]['是否评分为痴呆'][i] == '是' and df_sum[k]['是否诊断为痴呆'][i] == '是'):  # TP
                tmp[3] += 1
        acc = (tmp[0] + tmp[3]) / len(df_sum[k])
        acc_mmse.append('%.5f' % acc)

    col = ['文盲', '小学', '中学', '大学']
    # 取出用于预测的变量
    df_y = []
    df_X = []
    for ind in range(4):
        df_sum[ind].drop('学历', axis=1, inplace=True)
        df_sum[ind].drop('是否评分为痴呆', axis=1, inplace=True)
        df_sum[ind].drop('总分', axis=1, inplace=True)  # 删除df中MMSE,文化程度,入院诊断列
        #df_sum[ind].drop('4计算',axis = 1,inplace = True)
        #df_sum[ind].drop('2空间定向',axis = 1,inplace = True)
        #df_sum[ind].drop('11空间结构',axis = 1,inplace = True)
        df_tmp = df_sum[ind]['是否诊断为痴呆']  # 预测结果，其余为特征
        df_y.append(df_tmp)
        df_tmp = df_sum[ind].drop('是否诊断为痴呆', axis=1)
        df_X.append(df_tmp)

    # 创建Logistic回归模型
    print("正在训练逻辑回归模型...")
    scores_logreg = []
    for ind in range(4):
        logreg = LogisticRegression()
        ROC(df_X[ind], df_y[ind], logreg, '逻辑回归', col[ind])
        logreg.fit(df_X[ind], df_y[ind])
        scores = cross_val_score(logreg, df_X[ind], df_y[ind])
        scores_logreg.append('%.5f' % scores.mean())

    # 创建SVM模型
    print("正在训练SVM模型...")
    scores_svm = []
    for ind in range(4):
        svm = SVC(probability=True, random_state=1962, tol=1e-6)
        ROC(df_X[ind], df_y[ind], svm, 'SVM', col[ind])
        svm.fit(df_X[ind], df_y[ind])
        scores = cross_val_score(svm, df_X[ind], df_y[ind])
        scores_svm.append('%.5f' % scores.mean())

    # 创建决策树模型
    print("正在训练决策树模型...")
    scores_tree = []
    for ind in range(4):
        tree = DecisionTreeClassifier(random_state=1962)
        ROC(df_X[ind], df_y[ind], tree, '决策树', col[ind])
        tree.fit(df_X[ind], df_y[ind])
        scores = cross_val_score(tree, df_X[ind], df_y[ind])
        scores_tree.append('%.5f' % scores.mean())

    # 创建随机森林模型
    print("正在训练随机森林模型...")
    scores_forest = []
    for ind in range(4):
        forest = RandomForestClassifier(n_estimators=100, random_state=1962)
        ROC(df_X[ind], df_y[ind], forest, '随机森林', col[ind])
        forest.fit(df_X[ind], df_y[ind])
        scores = cross_val_score(forest, df_X[ind], df_y[ind])
        scores_forest.append('%.5f' % scores.mean())

    # 创建GBDT模型
    print("正在训练GBDT模型...")
    scores_GBDT = []
    for ind in range(4):
        Gbdt = GradientBoostingClassifier(random_state=1962)
        ROC(df_X[ind], df_y[ind], Gbdt, 'GBDT', col[ind])
        Gbdt.fit(df_X[ind], df_y[ind])
        scores = cross_val_score(Gbdt, df_X[ind], df_y[ind])
        scores_GBDT.append('%.5f' % scores.mean())

    # 回归结果可视化
    print("正在生成结果表格...")
    demo_df = pd.DataFrame({"MMSE评定": acc_mmse,
                            '逻辑回归': scores_logreg,
                            'SVM': scores_svm,
                            '决策树': scores_tree,
                            '随机森林': scores_forest,
                            'GBDT': scores_GBDT},
                           index=col)

    # 保存结果到输出目录
    output_path = os.path.join(output_dir, "结果.xlsx")
    demo_df.to_excel(output_path)
    print(f"结果已保存到: {output_path}")
    
    # 移动ROC图片到输出目录
    print("正在移动ROC图片...")
    for filename in os.listdir(data_dir):
        if filename.endswith('ROC.png'):
            src_path = os.path.join(data_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            os.rename(src_path, dst_path)
            print(f"已移动: {filename}")
    
    print("所有分析已完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行机器学习模型分析')
    parser.add_argument('--data-dir', '-d', type=str, default='../data',
                       help='数据文件目录路径 (默认: ../data)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='输出结果目录路径 (默认: 与数据目录相同)')
    
    args = parser.parse_args()
    
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出目录: {args.output_dir}")
    
    main(args.data_dir, args.output_dir)