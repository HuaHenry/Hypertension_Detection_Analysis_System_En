import pandas as pd
import joblib

def test_model(model_path, test_data_path):
    """
    加载保存的模型并对测试数据进行预测。
    输出预测的 Risk 和对应的置信度。
    """
    # 加载保存的最佳模型
    model = joblib.load(model_path)
    
    # 加载测试数据
    df_test = pd.read_csv(test_data_path)
    
    # 如果数据中包含目标变量，则移除
    if 'Risk' in df_test.columns:
        X_test = df_test.drop('Risk', axis=1)
    else:
        X_test = df_test
    
    # 确保分类变量转换为数值类型（若需要）
    for col in ['male', 'currentSmoker', 'BPMeds', 'diabetes']:
        if col in X_test.columns and X_test[col].dtype == object:
            mapping = {'Yes': 1, 'No': 0}
            X_test[col] = X_test[col].map(mapping)
    
    # 进行预测，得到预测标签和正类概率（置信度）
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # 构造结果 DataFrame 并输出
    results = pd.DataFrame({
        'Predicted_Risk': predictions,
        'Confidence': probabilities
    })
    
    print("预测结果：")
    print(results)
    
    # 保存预测结果到 CSV 文件中
    results.to_csv("test_predictions.csv", index=False)
    print("预测结果已保存为 'test_predictions.csv'")

if __name__ == "__main__":
    # 模型文件路径与测试数据路径（请根据实际情况修改路径）
    model_file = "best_heart_disease_model.pkl"
    test_data_file = "test_data.csv"
    
    test_model(model_file, test_data_file)