import pandas as pd
import joblib
import numpy as np

def predict_single_sample():
    """加载保存的模型并对单条数据进行预测"""
    # 加载模型
    try:
        model = joblib.load('best_heart_disease_model.pkl')
        print("模型加载成功!")
    except FileNotFoundError:
        print("错误: 找不到模型文件 'best_heart_disease_model.pkl'")
        return
    
    # 创建单条测试数据
    # 这里使用示例数据，你可以修改为你自己的数据
    test_data = {
        'age': 50,
        'education': 2,
        'male': 1,  # 1代表男性，0代表女性
        'currentSmoker': 0,  # 0代表不吸烟，1代表吸烟
        'cigsPerDay': 0,
        'BPMeds': 0,  # 0代表不服用血压药物，1代表服用
        'prevalentStroke': 0,
        'prevalentHyp': 0,
        'diabetes': 0,  # 0代表无糖尿病，1代表有糖尿病
        'totChol': 195,
        'sysBP': 130,
        'diaBP': 80,
        'BMI': 25.5,
        'heartRate': 70,
        'glucose': 85
    }
    
    # 将字典转换为DataFrame
    df = pd.DataFrame([test_data])
    print("\n输入数据:")
    print(df)
    
    # 进行预测
    try:
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        
        print("\n预测结果:")
        print(f"风险类别: {prediction[0]} (0: 低风险, 1: 高风险)")
        print(f"风险概率: {probability[0]:.4f}")
        
        # 保存结果到CSV文件
        result_df = pd.DataFrame({
            'feature': list(test_data.keys()),
            'value': list(test_data.values())
        })
        result_df.to_csv('single_sample_features.csv', index=False)
        
        pd.DataFrame({
            'prediction': prediction,
            'probability': probability
        }).to_csv('single_sample_prediction.csv', index=False)
        
        print("\n结果已保存到 'single_sample_prediction.csv'")
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")

def predict_custom_input():
    """让用户输入自定义数据进行预测"""
    # 加载模型
    try:
        model = joblib.load('best_heart_disease_model.pkl')
        print("模型加载成功!")
    except FileNotFoundError:
        print("错误: 找不到模型文件 'best_heart_disease_model.pkl'")
        return
    
    print("\n请输入患者数据:")
    try:
        age = int(input("年龄: "))
        education = int(input("教育水平 (1-4): "))
        male = int(input("性别 (1:男, 0:女): "))
        currentSmoker = int(input("当前是否吸烟 (1:是, 0:否): "))
        cigsPerDay = int(input("每天吸烟数量: ")) if currentSmoker == 1 else 0
        BPMeds = int(input("是否服用血压药物 (1:是, 0:否): "))
        prevalentStroke = int(input("既往是否有中风 (1:是, 0:否): "))
        prevalentHyp = int(input("是否患有高血压 (1:是, 0:否): "))
        diabetes = int(input("是否患有糖尿病 (1:是, 0:否): "))
        totChol = float(input("总胆固醇水平: "))
        sysBP = float(input("收缩压: "))
        diaBP = float(input("舒张压: "))
        BMI = float(input("BMI指数: "))
        heartRate = float(input("心率: "))
        glucose = float(input("血糖水平: "))
    except ValueError:
        print("输入错误，请确保输入正确的数据类型")
        return
    
    # 构建数据字典
    custom_data = {
        'age': age,
        'education': education,
        'male': male,
        'currentSmoker': currentSmoker,
        'cigsPerDay': cigsPerDay,
        'BPMeds': BPMeds,
        'prevalentStroke': prevalentStroke,
        'prevalentHyp': prevalentHyp,
        'diabetes': diabetes,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heartRate,
        'glucose': glucose
    }
    
    # 将字典转换为DataFrame
    df = pd.DataFrame([custom_data])
    
    # 进行预测
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    
    print("\n预测结果:")
    print(f"风险类别: {prediction[0]} (0: 低风险, 1: 高风险)")
    print(f"风险概率: {probability[0]:.4f}")
    
    # 解释风险水平
    if probability[0] < 0.3:
        risk_level = "低风险"
    elif probability[0] < 0.6:
        risk_level = "中等风险"
    else:
        risk_level = "高风险"
    
    print(f"风险评估: {risk_level}")

if __name__ == "__main__":
    print("心脏病风险预测测试程序")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 使用预设样本数据进行测试")
        print("2. 输入自定义数据进行预测")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ")
        
        if choice == '1':
            predict_single_sample()
        elif choice == '2':
            predict_custom_input()
        elif choice == '3':
            print("程序已退出")
            break
        else:
            print("无效选择，请输入1-3之间的数字")