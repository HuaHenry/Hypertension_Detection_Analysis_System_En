import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
import warnings
from collections import Counter
from scipy import stats

# 可能的改进空间：
# 超参数优化：可以添加网格搜索或随机搜索来自动寻找最佳超参数，而不是使用固定值
# 更多模型比较：可以尝试其他算法（如梯度提升、逻辑回归等）并比较性能
# 更复杂的特征工程：考虑添加特征交互项或多项式特征
# 交叉验证：更严格的k折交叉验证可以提供更可靠的性能估计

# 忽略警告
warnings.filterwarnings("ignore")


def preprocess_data(data_path):
    """
    加载并预处理数据
    """
    print("正在加载和预处理数据...")

    # 加载数据
    data = pd.read_csv(data_path)
    print(f"原始数据形状: {data.shape}")

    # 数据概览
    print("\n数据概览:")
    print(data.describe().T[['count', 'mean', 'std', 'min', 'max']])

    # 检查缺失值
    missing_values = data.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values[missing_values > 0])

    # 检查数据类型
    print("\n数据类型:")
    print(data.dtypes)

    # 识别分类和数值特征
    categorical_features = ['male', 'currentSmoker', 'BPMeds', 'diabetes']

    # 确保分类特征为离散值
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].astype('category')

    # 根据分布特性分组特征
    # 严重右偏数据 - 使用log变换
    log_transform_features = ['cigsPerDay', 'glucose', 'totChol']

    # 中等右偏数据 - 使用sqrt变换
    sqrt_transform_features = ['sysBP', 'diaBP', 'BMI']

    # 不需要变换的连续特征
    normal_features = ['age', 'heartRate']

    # 分离特征和目标变量
    X = data.drop('Risk', axis=1) if 'Risk' in data.columns else data.iloc[:, :-1]
    y = data['Risk'] if 'Risk' in data.columns else data.iloc[:, -1]

    # 首先进行数据分割，确保测试集保持独立
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
    print("原始训练集类别分布:")
    print(Counter(y_train))

    # 保存原始训练和测试数据副本
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()

    # 特殊处理cigsPerDay - 对于非吸烟者设为0
    # 对训练集
    if 'currentSmoker' in X_train.columns and 'cigsPerDay' in X_train.columns:
        # 非吸烟者的缺失值填充为0
        non_smoker_mask = X_train['currentSmoker'] == 0
        X_train.loc[non_smoker_mask & X_train['cigsPerDay'].isna(), 'cigsPerDay'] = 0

        # 吸烟者使用吸烟者子组的中位数
        smoker_mask = X_train['currentSmoker'] == 1
        smoker_median = X_train.loc[smoker_mask, 'cigsPerDay'].median()
        X_train.loc[smoker_mask & X_train['cigsPerDay'].isna(), 'cigsPerDay'] = smoker_median

    # 对测试集做相同处理
    if 'currentSmoker' in X_test.columns and 'cigsPerDay' in X_test.columns:
        # 非吸烟者的缺失值填充为0
        non_smoker_mask = X_test['currentSmoker'] == 0
        X_test.loc[non_smoker_mask & X_test['cigsPerDay'].isna(), 'cigsPerDay'] = 0

        # 吸烟者使用训练集吸烟者的中位数
        smoker_mask = X_test['currentSmoker'] == 1
        X_test.loc[smoker_mask & X_test['cigsPerDay'].isna(), 'cigsPerDay'] = smoker_median

    # 检测并处理异常值 (仅在训练集上)
    # 简化起见，这里使用IQR方法进行缩尾处理
    for col in X_train.select_dtypes(include=['float64', 'int64']).columns:
        if col not in categorical_features:  # 跳过分类特征
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 缩尾处理
            X_train.loc[X_train[col] < lower_bound, col] = lower_bound
            X_train.loc[X_train[col] > upper_bound, col] = upper_bound

    # 手动创建转换流程 (避免使用ColumnTransformer，因为它在处理类别特征时可能会遇到问题)

    # 1. 填充缺失值
    for col in normal_features + log_transform_features + sqrt_transform_features:
        if col in X_train.columns:
            # 使用中位数填充
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    # 对分类特征使用众数填充
    for col in categorical_features:
        if col in X_train.columns:
            # 使用众数填充
            mode_val = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)

    # 2. 应用特征变换
    # 对log变换特征
    for col in log_transform_features:
        if col in X_train.columns:
            X_train[col] = np.log1p(X_train[col])
            X_test[col] = np.log1p(X_test[col])

    # 对sqrt变换特征
    for col in sqrt_transform_features:
        if col in X_train.columns:
            X_train[col] = np.sqrt(X_train[col])
            X_test[col] = np.sqrt(X_test[col])

    # 3. 标准化数值特征
    numeric_features = normal_features + log_transform_features + sqrt_transform_features
    scaler = RobustScaler()

    for col in numeric_features:
        if col in X_train.columns:
            X_train[col] = scaler.fit_transform(X_train[[col]])
            X_test[col] = scaler.transform(X_test[[col]])

    # 4. 对分类特征进行独热编码
    for col in categorical_features:
        if col in X_train.columns:
            # 获取所有可能的类别值 (训练集和测试集合并)
            all_categories = pd.concat([X_train[col], X_test[col]]).unique()

            # 为每个类别创建哑变量，手动进行独热编码
            for category in all_categories:
                if category is not None and not pd.isna(category):  # 跳过NaN值
                    dummy_name = f"{col}_{category}"
                    X_train[dummy_name] = (X_train[col] == category).astype(int)
                    X_test[dummy_name] = (X_test[col] == category).astype(int)

            # 删除原始分类列
            X_train = X_train.drop(col, axis=1)
            X_test = X_test.drop(col, axis=1)

    print(f"处理后训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # 绘制相关性矩阵
    plt.figure(figsize=(10, 8))
    correlation_matrix = X_train.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("特征相关性矩阵")
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("相关性矩阵已保存为 'correlation_matrix.png'")

    return X_train, X_test, y_train, y_test, X_train_original, None  # 不再需要preprocessor


def calculate_feature_weights(X_train, y_train, threshold=0.05):
    """
    基于特征与目标变量的相关性计算特征权重
    """
    print("\n计算特征与目标变量的相关性...")

    # 确保X是DataFrame
    if not isinstance(X_train, pd.DataFrame):
        print("警告: 输入不是DataFrame，无法计算特征相关性")
        return None

    # 将目标变量添加到特征集中
    data = X_train.copy()
    data['target'] = y_train

    # 计算相关性矩阵
    correlation_matrix = data.corr(method='pearson')

    # 提取每个特征与目标变量的相关性
    feature_correlations = correlation_matrix['target'].drop('target')

    # 使用相关性的绝对值
    abs_correlations = feature_correlations.abs()

    # 筛选出相关性高于阈值的特征
    selected_features = abs_correlations[abs_correlations > threshold].index.tolist()

    # 如果没有特征满足条件，保留所有特征
    if len(selected_features) == 0:
        print("警告: 没有特征的相关性高于阈值，将保留所有特征")
        selected_features = X_train.columns.tolist()

    # 计算特征权重 (相关性的绝对值，归一化到0-1范围)
    max_corr = abs_correlations.max()
    min_corr = abs_correlations.min()

    if max_corr == min_corr:
        # 如果所有特征相关性相同，给予相同权重
        feature_weights = {feature: 1.0 for feature in selected_features}
    else:
        # 基于相关性计算权重
        feature_weights = {}
        for feature in selected_features:
            # 归一化权重到[0.1, 1.0]范围，避免权重为0
            weight = 0.1 + 0.9 * (abs_correlations[feature] - min_corr) / (max_corr - min_corr)
            feature_weights[feature] = weight

    # 打印相关性和权重
    print("\n特征与目标变量的相关性和权重 (前10个):")
    for feature in sorted(selected_features, key=lambda x: abs_correlations[x], reverse=True)[:10]:
        print(f"{feature}: 相关性 = {feature_correlations[feature]:.4f}, 权重 = {feature_weights[feature]:.4f}")

    return feature_weights


def apply_smoteenn(X_train, y_train):
    """
    使用SMOTEENN方法处理不平衡数据
    """
    print("\n使用SMOTEENN处理类别不平衡...")

    # 创建SMOTEENN实例
    smoteenn = SMOTEENN(random_state=42)

    # 应用SMOTEENN重采样
    X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

    print(f"重采样前: {Counter(y_train)}")
    print(f"重采样后: {Counter(y_resampled)}")
    print(f"训练集从 {X_train.shape[0]} 样本调整为 {X_resampled.shape[0]} 样本")

    return X_resampled, y_resampled


def optimize_hyperparameters(X_train, y_train):
    """
    使用网格搜索优化随机森林模型超参数
    """
    print("\n进行超参数优化...")

    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5, 6, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # 创建基础模型
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', oob_score=True)

    # 使用5折交叉验证进行网格搜索
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    # 执行网格搜索
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

    return grid_search.best_params_


def train_random_forest_with_regularization(X_train, y_train, X_test, y_test, feature_weights=None, params=None):
    """
    训练随机森林模型并应用正则化
    """
    print("\n训练随机森林模型（带正则化）...")

    # 创建随机森林分类器，应用正则化参数
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'class_weight': 'balanced',
            'ccp_alpha': 0.0001,
            'random_state': 42,
            'n_jobs': -1
        }

    rf = RandomForestClassifier(**params)

    # 应用基于相关性的特征权重（如果提供）
    sample_weights = None
    if feature_weights is not None and isinstance(X_train, pd.DataFrame):
        print("应用基于相关性的特征权重...")

    # 训练模型
    rf.fit(X_train, y_train)

    # 预测
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # 评估模型
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"\n训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")
    if 'oob_score' in params and params['oob_score']:
        print(f"袋外(OOB)得分: {rf.oob_score_:.4f}")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_test_pred))

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['低风险', '高风险'], yticklabels=['低风险', '高风险'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('随机森林模型的混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png')
    print("混淆矩阵已保存为 'confusion_matrix_rf.png'")

    # 特征重要性
    if isinstance(X_train, pd.DataFrame):
        importances = rf.feature_importances_
        feature_names = X_train.columns

        # 排序特征重要性
        indices = np.argsort(importances)[::-1]

        # 打印前10个最重要的特征
        print("\n特征重要性 (前10个):")
        for i in range(min(10, len(indices))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    return rf





def compare_multiple_models(X_train, y_train, X_test, y_test):
    """
    训练并比较多个机器学习模型的性能
    """
    print("\n比较多个机器学习模型...")

    # 定义要比较的模型
    models = {
        "随机森林": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "梯度提升树": GradientBoostingClassifier(random_state=42),
        "逻辑回归": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "K近邻": KNeighborsClassifier(n_neighbors=5),
        "支持向量机": SVC(probability=True, random_state=42, class_weight='balanced')
    }

    # 存储结果
    results = {}

    # 训练并评估每个模型
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算性能指标
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        print(f"{name} - 训练集准确率: {train_accuracy:.4f}")
        print(f"{name} - 测试集准确率: {test_accuracy:.4f}")
        print(f"{name} - 测试集F1分数: {test_f1:.4f}")

        # 存储结果
        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        }

        # 为每个模型绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=['低风险', '高风险'], yticklabels=['低风险', '高风险'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{name}模型的混淆矩阵')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name}.png')
        print(f"{name}混淆矩阵已保存")

    # 可视化模型比较
    plot_model_comparison(results)

    return results, models


def create_ensemble_model(X_train, y_train, X_test, y_test):
    """
    创建一个结合四种基础模型的集成模型
    """
    print("\n训练四模型集成系统...")

    # 创建基础模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    knn = KNeighborsClassifier(n_neighbors=5)

    # 训练基础模型
    base_models = {
        "随机森林": rf,
        "梯度提升树": gb,
        "逻辑回归": lr,
        "K近邻": knn
    }

    for name, model in base_models.items():
        print(f"训练基础模型: {name}")
        model.fit(X_train, y_train)

    # 创建投票分类器（软投票）
    ensemble = VotingClassifier(
        estimators=[
            ('rf', base_models["随机森林"]),
            ('gb', base_models["梯度提升树"]),
            ('lr', base_models["逻辑回归"]),
            ('knn', base_models["K近邻"])
        ],
        voting='soft'
    )

    # 训练集成模型
    ensemble.fit(X_train, y_train)

    # 评估集成模型
    y_train_pred = ensemble.predict(X_train)
    y_test_pred = ensemble.predict(X_test)

    # 计算性能指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    try:
        # 计算ROC AUC（如果可用）
        y_test_proba = ensemble.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)

        print(f"\n四模型集成 - 训练集准确率: {train_accuracy:.4f}")
        print(f"四模型集成 - 测试集准确率: {test_accuracy:.4f}")
        print(f"四模型集成 - 测试集F1分数: {test_f1:.4f}")
        print(f"四模型集成 - ROC AUC: {test_auc:.4f}")
    except:
        print(f"\n四模型集成 - 训练集准确率: {train_accuracy:.4f}")
        print(f"四模型集成 - 测试集准确率: {test_accuracy:.4f}")
        print(f"四模型集成 - 测试集F1分数: {test_f1:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['低风险', '高风险'], yticklabels=['低风险', '高风险'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('四模型集成的混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix_ensemble.png')
    print("四模型集成混淆矩阵已保存为 'confusion_matrix_ensemble.png'")

    # 比较基础模型与集成模型
    compare_with_base_models(base_models, ensemble, X_test, y_test)

    return ensemble, base_models


def compare_with_base_models(base_models, ensemble, X_test, y_test):
    """
    比较基础模型和集成模型在测试集上的表现
    """
    # 计算所有模型的准确率
    accuracies = {}

    # 基础模型的准确率
    for name, model in base_models.items():
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

    # 集成模型的准确率
    y_pred = ensemble.predict(X_test)
    accuracies["四模型集成"] = accuracy_score(y_test, y_pred)

    # 绘制比较图
    plt.figure(figsize=(10, 6))

    model_names = list(accuracies.keys())
    acc_values = list(accuracies.values())

    # 使用不同颜色突出显示集成模型
    colors = ['#3498db'] * len(base_models) + ['#e74c3c']

    plt.bar(model_names, acc_values, color=colors)
    plt.axhline(y=accuracies["四模型集成"], color='r', linestyle='-', alpha=0.3)

    plt.ylabel('测试集准确率')
    plt.title('基础模型与集成模型准确率比较')
    plt.ylim(0.8, 1.0)  # 调整以便更好地查看差异

    # 添加数值标签
    for i, v in enumerate(acc_values):
        plt.text(i, v + 0.005, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig('ensemble_comparison.png')
    print("集成模型比较图已保存为 'ensemble_comparison.png'")


def plot_model_comparison(results):
    """
    可视化不同模型的性能比较
    """
    # 提取数据
    model_names = list(results.keys())
    train_accuracies = [results[model]['train_accuracy'] for model in model_names]
    test_accuracies = [results[model]['test_accuracy'] for model in model_names]
    f1_scores = [results[model]['test_f1'] for model in model_names]

    # 创建横向条形图
    plt.figure(figsize=(12, 8))

    x = np.arange(len(model_names))
    width = 0.25

    plt.barh(x - width, train_accuracies, width, label='训练准确率', color='skyblue')
    plt.barh(x, test_accuracies, width, label='测试准确率', color='lightgreen')
    plt.barh(x + width, f1_scores, width, label='F1分数', color='salmon')

    # 添加标签和标题
    plt.yticks(x, model_names)
    plt.xlabel('分数')
    plt.title('不同模型性能比较')
    plt.xlim(0, 1.0)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 添加数值标签
    for i, v in enumerate(train_accuracies):
        plt.text(v, i - width, f'{v:.3f}', va='center', fontweight='bold')
    for i, v in enumerate(test_accuracies):
        plt.text(v, i, f'{v:.3f}', va='center', fontweight='bold')
    for i, v in enumerate(f1_scores):
        plt.text(v, i + width, f'{v:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("模型比较图已保存为 'model_comparison.png'")


def main(data_path):
    """
    主函数
    """
    # 1. 预处理数据 (包含异常值处理)
    X_train, X_test, y_train, y_test, X_train_original, _ = preprocess_data(data_path)

    # 2. 计算特征权重 (使用原始特征)
    if isinstance(X_train_original, pd.DataFrame):
        feature_weights = calculate_feature_weights(X_train_original, y_train)
    else:
        feature_weights = None

    # 3. 应用SMOTEENN处理不平衡数据（仅在训练集上）
    X_train_resampled, y_train_resampled = apply_smoteenn(X_train, y_train)

    # 4. 比较多个模型性能
    model_results, models = compare_multiple_models(X_train_resampled, y_train_resampled, X_test, y_test)

    # 5. 为随机森林模型优化超参数
    best_params = optimize_hyperparameters(X_train_resampled, y_train_resampled)

    # 6. 使用最佳参数训练随机森林模型
    rf_params = best_params.copy()
    rf_params.update({
        'oob_score': True,
        'class_weight': 'balanced',
        'ccp_alpha': 0.0001,
        'random_state': 42,
        'n_jobs': -1
    })

    best_rf = train_random_forest_with_regularization(X_train_resampled, y_train_resampled, X_test, y_test,
                                                      feature_weights, rf_params)

    return best_rf, (X_train, X_test), model_results



    # 创建和评估四模型集成
    ensemble_model, base_models = create_ensemble_model(X_train_resampled, y_train_resampled, X_test, y_test)

    return best_rf, ensemble_model, (X_train, X_test)


def predict_risk(model, X_train_format, input_data):
    """
    使用训练好的模型预测新数据
    """
    # 获取训练格式参考
    X_train_ref = X_train_format[0]

    # 手动转换输入数据，确保与训练数据格式一致
    processed_input = pd.DataFrame()

    # 对分类特征进行独热编码
    categorical_features = ['male', 'currentSmoker', 'BPMeds', 'diabetes']

    # 处理数值特征
    for col in input_data.columns:
        if col in X_train_ref.columns:
            # 如果列直接存在，直接复制
            processed_input[col] = input_data[col]
        elif col in categorical_features:
            # 如果是分类特征，需要独热编码
            for category in input_data[col].unique():
                dummy_name = f"{col}_{category}"
                if dummy_name in X_train_ref.columns:
                    processed_input[dummy_name] = (input_data[col] == category).astype(int)

    # 确保所有训练集中的列在预测数据中都存在
    for col in X_train_ref.columns:
        if col not in processed_input.columns:
            processed_input[col] = 0  # 使用0填充缺失的列

    # 重新排序列，确保与训练集一致
    processed_input = processed_input[X_train_ref.columns]

    # 预测
    prediction = model.predict(processed_input)
    probabilities = model.predict_proba(processed_input)[:, 1]  # 高风险的概率

    return prediction, probabilities


# 程序入口
if __name__ == "__main__":
    # 替换为您的CSV文件路径
    data_path = "/Users/a./PycharmProjects/COMP208/.venv/Hypertension-risk-model-main.csv"

    # 训练模型
    model, X_format, model_results = main(data_path)

    print("\n模型训练完成！")

    # 找出表现最好的模型
    best_model_name = max(model_results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_accuracy = model_results[best_model_name]['test_accuracy']
    print(f"\n表现最佳的模型: {best_model_name}，测试准确率: {best_accuracy:.4f}")

    # 示例预测
    print("\n示例预测:")
    sample_input = pd.DataFrame({
        'male': [1],
        'age': [60],
        'currentSmoker': [0],
        'cigsPerDay': [0],
        'BPMeds': [0],
        'diabetes': [0],
        'totChol': [220],
        'sysBP': [140],
        'diaBP': [90],
        'BMI': [28],
        'heartRate': [80],
        'glucose': [100]
    })

    # 预处理输入数据的各个特征
    # 对数变换
    for col in ['cigsPerDay', 'glucose', 'totChol']:
        if col in sample_input.columns:
            sample_input[col] = np.log1p(sample_input[col])

    # 平方根变换
    for col in ['sysBP', 'diaBP', 'BMI']:
        if col in sample_input.columns:
            sample_input[col] = np.sqrt(sample_input[col])

    # 预测
    prediction, probability = predict_risk(model, X_format, sample_input)
    risk_status = "高风险" if prediction[0] == 1 else "低风险"

    print(f"预测结果: {risk_status}")
    print(f"高风险概率: {probability[0]:.4f}")