from flask import Flask, render_template, request
from flask import jsonify
from flask import redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import re
from flask_migrate import Migrate
from enum import Enum
from sqlalchemy import desc
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

import csv
from io import StringIO
from flask import make_response

from datetime import datetime

import joblib

import json
from flask import request, jsonify, render_template
from flask_login import login_required, current_user
import requests
from flask import Flask, render_template, request, jsonify, make_response
from flask_login import login_required, current_user

model = joblib.load('ML/best_heart_disease_model.pkl')

from openai import OpenAI

class GenderEnum(Enum):
    MALE = 'male'
    FEMALE = 'female'
    OTHER = 'other'

    @classmethod
    def get_by_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        return None

class BloodTypeEnum(Enum):
    A_POSITIVE = 'A+'
    A_NEGATIVE = 'A-'
    B_POSITIVE = 'B+'
    B_NEGATIVE = 'B-'
    AB_POSITIVE = 'AB+'
    AB_NEGATIVE = 'AB-'
    O_POSITIVE = 'O+'
    O_NEGATIVE = 'O-'


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db) 



login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    # 基本账户信息
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # 新增个人信息
    age = db.Column(db.Integer)
    gender = db.Column(db.Enum(GenderEnum))
    height = db.Column(db.Float)  # 单位: cm
    weight = db.Column(db.Float)  # 单位: kg
    blood_type = db.Column(db.Enum(BloodTypeEnum))
    
    # 健康相关指标
    has_hypertension = db.Column(db.Boolean, default=False)
    has_diabetes = db.Column(db.Boolean, default=False)
    is_smoker = db.Column(db.Boolean, default=False)
    family_history = db.Column(db.String(500))  # 家族病史
    
    # Flask-Login required properties
    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_anonymous(self):
        return False
    
    # 计算属性
    @property
    def bmi(self):
        if self.height and self.weight:
            return round(self.weight / ((self.height/100) ** 2), 1)
        return None
    
    # Password handling
    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def get_gender_display(self):
        if not self.gender:
            return ""
        return self.gender.value.capitalize()  # 返回"Male", "Female", "Other"

    def get_blood_type_display(self):
        if not self.blood_type:
            return ""
        return self.blood_type.value  # 返回"A+", "B-"等
    
class BloodPressureRecord(db.Model):
    __tablename__ = 'blood_pressure_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    systolic = db.Column(db.Integer, nullable=False)  # 收缩压
    diastolic = db.Column(db.Integer, nullable=False)  # 舒张压
    pulse = db.Column(db.Integer)  # 脉搏
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.String(500))  # 备注
    
    # 新增机器学习预测相关字段
    predicted_risk = db.Column(db.Float)  # 预测风险值(0-1)
    predicted_status = db.Column(db.String(20))  # 预测状态
    
    # 与用户的关系
    user = db.relationship('User', backref=db.backref('blood_pressure_records', lazy=True))
    
    # 计算血压状态(基于您的ML算法)
    @property
    def status(self):
        if not self.predicted_status:
            # 如果没有预测状态，使用默认分类
            return self._default_status()
        return self.predicted_status
    
    # 获取状态对应的CSS类
    @property
    def status_class(self):
        status = self.status.lower()
        if status == "low risk":
            return "bg-success"
        elif status == "elevated":
            return "bg-warning"
        elif "stage" in status:
            return "bg-danger"
        else:
            return "bg-info"
    
    def _default_status(self):
        """在没有ML预测时的默认分类"""
        if self.systolic < 90 or self.diastolic < 60:
            return "Hypotension"
        elif self.systolic < 120 and self.diastolic < 80:
            return "Normal"
        elif (120 <= self.systolic < 130) and self.diastolic < 80:
            return "Elevated"
        elif (130 <= self.systolic < 140) or (80 <= self.diastolic < 90):
            return "Hypertension Stage 1"
        elif (140 <= self.systolic < 180) or (90 <= self.diastolic < 120):
            return "Hypertension Stage 2"
        else:
            return "Hypertensive Crisis"
    
    def __repr__(self):
        return f'<BloodPressureRecord {self.systolic}/{self.diastolic} mmHg>'

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/')
@login_required
def index():
    latest_record = BloodPressureRecord.query.filter_by(user_id=current_user.id)\
                        .order_by(desc(BloodPressureRecord.recorded_at)).first()
    return render_template('index.html', latest_record=latest_record)

@app.route('/add_record', methods=['GET', 'POST'])
@login_required
def add_record():
    if request.method == 'POST':
        try:
            systolic = int(request.form.get('systolic'))
            diastolic = int(request.form.get('diastolic'))
            pulse = int(request.form.get('pulse', 0))
            notes = request.form.get('notes', '')
            
            # 验证数据
            if systolic < 50 or systolic > 250:
                flash('Systolic pressure must be between 50-250 mmHg', 'danger')
                return redirect(url_for('add_record'))
            
            if diastolic < 30 or diastolic > 150:
                flash('Diastolic pressure must be between 30-150 mmHg', 'danger')
                return redirect(url_for('add_record'))
            
            if pulse and (pulse < 30 or pulse > 200):
                flash('Pulse must be between 30-200 bpm', 'danger')
                return redirect(url_for('add_record'))
            
            # 调用ML模型预测风险
            prediction, risk_score = predict_blood_pressure_risk(
                current_user, systolic, diastolic, pulse
            )
            
            # 创建新记录
            new_record = BloodPressureRecord(
                user_id=current_user.id,
                systolic=systolic,
                diastolic=diastolic,
                pulse=pulse,
                notes=notes,
                predicted_risk=risk_score,
                predicted_status=prediction
            )
            
            db.session.add(new_record)
            db.session.commit()
            
            flash('Blood pressure record added successfully!', 'success')
            return redirect(url_for('index'))
            
        except ValueError:
            flash('Please enter valid numbers for all fields', 'danger')
            return redirect(url_for('add_record'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding record: {str(e)}', 'danger')
            return redirect(url_for('add_record'))
    
    return render_template('add_record.html')

# def predict_blood_pressure_risk(user, systolic, diastolic, pulse):
#     """
#     调用您的ML模型预测血压风险
#     这里应该是您提供的ML算法的接口
#     """
#     # 这里应该是调用您提供的ML模型的代码
#     # 示例实现 - 请替换为您的实际模型调用
    
#     # 构建特征数据框
#     input_data = pd.DataFrame({
#         'male': [1 if user.gender == GenderEnum.MALE else 0],
#         'age': [user.age or 50],  # 默认值以防用户没有设置年龄
#         'currentSmoker': [1 if user.is_smoker else 0],
#         'cigsPerDay': [0],  # 需要根据用户数据设置
#         'BPMeds': [1 if user.has_hypertension else 0],
#         'diabetes': [1 if user.has_diabetes else 0],
#         'totChol': [200],  # 需要实际数据
#         'sysBP': [systolic],
#         'diaBP': [diastolic],
#         'BMI': [user.bmi or 25],  # 使用用户BMI或默认值
#         'heartRate': [pulse or 72],  # 使用脉搏或默认值
#         'glucose': [100]  # 需要实际数据
#     })
    
#     # 这里应该调用您的预处理和预测流程
#     # prediction, probability = your_ml_model.predict(input_data)
    
#     # 示例返回值 - 替换为实际模型输出
#     risk_score = random.uniform(0, 1)  # 示例随机值
#     if risk_score < 0.3:
#         prediction = "Low risk"
#     elif risk_score < 0.6:
#         prediction = "Medium risk"
#     else:
#         prediction = "High risk"
    
#     return prediction, risk_score

# def predict_blood_pressure_risk(user, systolic, diastolic, pulse):
#     """
#     Use the trained machine learning model to predict blood pressure risk.
#     """
#     # Construct the feature data
#     input_data = pd.DataFrame({
#         'male': [1 if user.gender == GenderEnum.MALE else 0],
#         'age': [user.age or 50],
#         'currentSmoker': [1 if user.is_smoker else 0],
#         'cigsPerDay': [0],  # Adjust if needed based on user data
#         'BPMeds': [1 if user.has_hypertension else 0],
#         'diabetes': [1 if user.has_diabetes else 0],
#         'totChol': [200],   # Example value, adjust as necessary
#         'sysBP': [systolic],
#         'diaBP': [diastolic],
#         'BMI': [user.bmi or 25],
#         'heartRate': [pulse or 72],
#         'glucose': [100]    # Example value
#     })
    
#     # Align input_data with the training format (train_format)
#     processed_input = pd.DataFrame()
#     for col in train_format.columns:
#         if col in input_data.columns:
#             processed_input[col] = input_data[col]
#         else:
#             processed_input[col] = 0
#     processed_input = processed_input[train_format.columns]
    
#     # Make prediction
#     prediction = model.predict(processed_input)
#     probability = model.predict_proba(processed_input)[:, 1]  # Assume label 1 indicates high risk
    
#     # Determine risk status
#     if prediction[0] == 1:
#         risk_status = "High risk"
#     elif probability[0] > 0.4:
#         risk_status = "Medium risk"
#     else:
#         risk_status = "Low risk"
    
#     return risk_status, float(probability[0])

def predict_blood_pressure_risk(user, systolic, diastolic, pulse):
    """
    使用训练好的模型预测血压风险，返回预测的 Risk（0 或 1）及其置信度（正类概率）。
    """
    # 构造输入特征，假设模型训练时使用的特征顺序为：
    # ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes',
    #  'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    input_data = pd.DataFrame({
        'male': [1 if user.gender == GenderEnum.MALE else 0],
        'age': [user.age or 50],
        'currentSmoker': [1 if user.is_smoker else 0],
        'cigsPerDay': [getattr(user, 'cigsPerDay', 0)],
        'BPMeds': [1 if user.has_hypertension else 0],
        'diabetes': [1 if user.has_diabetes else 0],
        'totChol': [getattr(user, 'totChol', 200)],   # 如有数据可调整
        'sysBP': [systolic],
        'diaBP': [diastolic],
        'BMI': [user.bmi or 25],
        'heartRate': [pulse or 72],
        'glucose': [getattr(user, 'glucose', 100)]      # 如有数据可调整
    })
    
    # 模型预测：获取预测标签和正类概率（假设 Risk=1 表示高风险）
    predicted_risk = "High Risk" if model.predict(input_data)[0] == 1 else "Low Risk"
    confidence = model.predict_proba(input_data)[0][1]
    
    return str(predicted_risk), float(confidence)

@app.route('/history')
@login_required
def history():
    # 获取当前用户的所有血压记录，按时间降序排列
    records = BloodPressureRecord.query.filter_by(user_id=current_user.id)\
                    .order_by(desc(BloodPressureRecord.recorded_at)).all()
    
    if records:
        # 准备图表数据
        dates = [record.recorded_at.strftime('%Y-%m-%d %H:%M') for record in records]
        systolic_values = [record.systolic for record in records]
        diastolic_values = [record.diastolic for record in records]
        pulse_values = [record.pulse for record in records] if any(record.pulse for record in records) else None
        
        # 计算状态值 (1: Normal, 2: Elevated, 3: High)
        status_values = []
        for record in records:
            # if record.systolic < 120 and record.diastolic < 80:
            #     status_values.append(1)  # Normal
            # elif (120 <= record.systolic < 140) or (80 <= record.diastolic < 90):
            #     status_values.append(2)  # Elevated
            # else:
            #     status_values.append(3)  # High
            status_values
        
        # 计算统计数据
        avg_systolic = sum(systolic_values) / len(systolic_values)
        max_systolic = max(systolic_values)
        avg_diastolic = sum(diastolic_values) / len(diastolic_values)
        max_diastolic = max(diastolic_values)
        
        if pulse_values:
            avg_pulse = sum(pulse_values) / len(pulse_values)
            max_pulse = max(pulse_values)
        else:
            avg_pulse = None
            max_pulse = None
        
        return render_template('history.html',
                           records=records,
                           dates=dates,
                           systolic_values=systolic_values,
                           diastolic_values=diastolic_values,
                           pulse_values=pulse_values or [],
                           status_values=status_values,
                           avg_systolic=avg_systolic,
                           max_systolic=max_systolic,
                           avg_diastolic=avg_diastolic,
                           max_diastolic=max_diastolic,
                           avg_pulse=avg_pulse,
                           max_pulse=max_pulse)
    else:
        return render_template('history.html', records=None)

@app.route('/health_knowledge')
def health_knowledge():
    return render_template('health_knowledge.html')

# @app.route('/personal_center')
# def personal_center():
#     return render_template('personal_center.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def personal_center():
    return render_template('profile.html', user=current_user)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    # 基本账户信息
    username = request.form.get('username')
    email = request.form.get('email')
    
    # 验证用户名是否已存在
    if username != current_user.username:
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('personal_center'))
    
    # 验证邮箱是否已存在
    if email != current_user.email:
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered', 'danger')
            return redirect(url_for('personal_center'))
    
    # 更新个人信息
    try:
        current_user.username = username
        current_user.email = email
        current_user.age = request.form.get('age', type=int)
        
        # 处理性别 - 使用新方法
        gender = request.form.get('gender')
        if gender:
            current_user.gender = GenderEnum.get_by_value(gender)
            
        # 处理身高体重
        current_user.height = request.form.get('height', type=float)
        current_user.weight = request.form.get('weight', type=float)
        
        # 处理血型
        blood_type = request.form.get('bloodType')
        if blood_type:
            current_user.blood_type = BloodTypeEnum(blood_type)
        
        db.session.commit()
        flash('Profile updated successfully', 'success')
    except ValueError as e:
        db.session.rollback()
        flash(f'Invalid value: {str(e)}', 'danger')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating profile: {str(e)}', 'danger')
    
    return redirect(url_for('personal_center'))

#  登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.verify_password(password):
            if user.is_active:  # 检查用户是否激活
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Your account is disabled', 'warning')
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        terms = request.form.get('terms')
        
        # 验证逻辑
        errors = []
        
        # 用户名验证
        if not username or len(username) < 4 or len(username) > 20:
            errors.append('Username must be between 4-20 characters')
        elif not re.match(r'^[A-Za-z0-9_]+$', username):
            errors.append('Username can only contain letters, numbers and underscores')
        elif User.query.filter_by(username=username).first():
            errors.append('Username already exists')
        
        # 邮箱验证
        if not email or '@' not in email:
            errors.append('Invalid email address')
        elif User.query.filter_by(email=email).first():
            errors.append('Email already registered')
        
        # 密码验证
        if not password or len(password) < 8:
            errors.append('Password must be at least 8 characters')
        elif password != confirm_password:
            errors.append('Passwords do not match')
        
        # 条款验证
        if not terms:
            errors.append('You must accept the terms of service')
        
        # 如果有错误，显示并返回
        if errors:
            for error in errors:
                flash(error, 'danger')
            return redirect(url_for('register'))
        
        # 创建新用户
        try:
            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password, method='pbkdf2:sha256'),
                is_active=True  # 默认激活账户
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/data_analysis')
@login_required
def data_analysis():
    # 查询当前用户的所有血压记录（按记录时间升序排列）
    records = BloodPressureRecord.query.filter_by(user_id=current_user.id)\
                .order_by(BloodPressureRecord.recorded_at.asc()).all()
    
    formatted_records = []
    if records:
        total_records = len(records)
        systolic_values = []
        diastolic_values = []
        trend_labels = []
        for record in records:
            # 假设 BloodPressureRecord.recorded_at 为 datetime 对象
            formatted_records.append({
                'date': record.recorded_at.strftime('%Y-%m-%d'),
                'time': record.recorded_at.strftime('%H:%M'),
                'systolic': record.systolic,
                'diastolic': record.diastolic,
                'status': record.status  # 状态应在保存记录时设置好，如 '正常', '偏高', '高'
            })
            systolic_values.append(record.systolic)
            diastolic_values.append(record.diastolic)
            trend_labels.append(record.recorded_at.strftime('%b %d'))
        
        avg_systolic = sum(systolic_values) / total_records
        avg_diastolic = sum(diastolic_values) / total_records
        last_record_date = records[-1].recorded_at.strftime('%Y-%m-%d %H:%M')
    else:
        total_records = 0
        avg_systolic = 0
        avg_diastolic = 0
        last_record_date = 'N/A'
        trend_labels = []
        systolic_values = []
        diastolic_values = []
    
    return render_template("data_analysis.html",
                           records=formatted_records,
                           avg_systolic=avg_systolic,
                           avg_diastolic=avg_diastolic,
                           total_records=total_records,
                           last_record_date=last_record_date,
                           trend_labels=trend_labels,
                           trend_systolic=systolic_values,
                           trend_diastolic=diastolic_values)

# @app.route('/data_analysis')
# def data_analysis():
#     # 示例数据
#     # 后端数据库接入后修改
#     records = [
#         {"date": "2025/3/10", "time": "23:50", "systolic": 120, "diastolic": 80, "status": "正常"},
#         {"date": "2025/3/11", "time": "01:04", "systolic": 130, "diastolic": 90, "status": "偏高"},
#         {"date": "2025/3/12", "time": "22:30", "systolic": 125, "diastolic": 85, "status": "正常"},
#         {"date": "2025/3/13", "time": "23:15", "systolic": 135, "diastolic": 95, "status": "高"},
#         {"date": "2025/3/14", "time": "00:45", "systolic": 128, "diastolic": 88, "status": "偏高"},
#     ]
#     return render_template('data_analysis.html', records=records)

@app.route('/export_csv')
@login_required
def export_csv():
    # 查询当前用户的所有血压记录（按记录时间升序排列）
    records = BloodPressureRecord.query.filter_by(user_id=current_user.id)\
                .order_by(BloodPressureRecord.recorded_at.asc()).all()
    
    # 使用 StringIO 和 csv 模块生成 CSV 内容
    si = StringIO()
    cw = csv.writer(si)
    
    # 写入表头
    cw.writerow(["Date", "Time", "Systolic", "Diastolic", "Status"])
    
    # 写入每条记录（假设 BloodPressureRecord.recorded_at 为 datetime 对象）
    for record in records:
        cw.writerow([
            record.recorded_at.strftime("%Y-%m-%d"),
            record.recorded_at.strftime("%H:%M"),
            record.systolic,
            record.diastolic,
            record.status
        ])
    
    # 获取 CSV 内容
    output = si.getvalue()
    
    # 创建响应，并设置响应头以实现下载 CSV 文件
    response = make_response(output)
    response.headers["Content-Disposition"] = "attachment; filename=bp_records.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route('/blood_pressure_assessment', methods=['GET', 'POST'])
def blood_pressure_assessment():
    """Render the blood pressure assessment form"""
    return render_template('blood_pressure_assessment.html')

@app.route('/api/assess_blood_pressure', methods=['POST'])
def assess_blood_pressure():
    """API endpoint for blood pressure assessment"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'age', 'male', 'currentSmoker', 'BPMeds', 'diabetes',
            'sysBP', 'diaBP', 'totChol', 'BMI', 'heartRate', 'glucose'
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Convert and validate numeric fields
        try:
            age = int(data['age'])
            sysBP = int(data['sysBP'])
            diaBP = int(data['diaBP'])
            totChol = int(data['totChol'])
            BMI = float(data['BMI'])
            heartRate = int(data['heartRate'])
            glucose = int(data['glucose'])
            cigsPerDay = int(data.get('cigsPerDay', 0))
            
            if age < 18 or age > 120:
                raise ValueError("Age must be between 18 and 120")
            if sysBP < 70 or sysBP > 250:
                raise ValueError("Systolic BP must be between 70 and 250 mmHg")
            if diaBP < 40 or diaBP > 150:
                raise ValueError("Diastolic BP must be between 40 and 150 mmHg")
            if totChol < 100 or totChol > 600:
                raise ValueError("Total cholesterol must be between 100 and 600 mg/dL")
            if BMI < 15 or BMI > 50:
                raise ValueError("BMI must be between 15 and 50 kg/m²")
            if heartRate < 40 or heartRate > 200:
                raise ValueError("Heart rate must be between 40 and 200 bpm")
            if glucose < 50 or glucose > 400:
                raise ValueError("Glucose must be between 50 and 400 mg/dL")
            if cigsPerDay < 0 or cigsPerDay > 100:
                raise ValueError("Cigarettes per day must be between 0 and 100")
                
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400

        # Your assessment logic here
        status, risk_score = calculate_risk_score(
            age=age,
            male=data['male'] == '1',
            currentSmoker=data['currentSmoker'] == '1',
            cigsPerDay=cigsPerDay,
            BPMeds=data['BPMeds'] == '1',
            diabetes=data['diabetes'] == '1',
            sysBP=sysBP,
            diaBP=diaBP,
            totChol=totChol,
            BMI=BMI,
            heartRate=heartRate,
            glucose=glucose
        )

        if status == "High Risk":
            result = {
                'status': 'warning',
                'message': 'Your blood pressure is high. Please consult a doctor.',
                'details': {
                    'risk_score': risk_score,
                }
            }
        else:
            result = {
                'status': 'success',
                'message': 'Your blood pressure is normal. Keep maintaining a healthy lifestyle.',
                'details': {
                    'risk_score': risk_score,
                }
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500

# def calculate_risk_score(**kwargs):
#     """Example risk score calculation - replace with your actual algorithm"""
#     risk = 0
    
#     # Simple example calculation (replace with your actual algorithm)
#     if kwargs['sysBP'] >= 140: risk += 2
#     if kwargs['diaBP'] >= 90: risk += 2
#     if kwargs['currentSmoker']: risk += 1
#     if kwargs['diabetes']: risk += 1
#     if kwargs['BPMeds']: risk += 1
#     if kwargs['age'] > 50: risk += 1
#     if kwargs['BMI'] > 30: risk += 1
    
#     return f"{risk}/8 risk factors"  # Example format

import pandas as pd

def calculate_risk_score(**kwargs):
    """
    使用训练好的模型预测血压风险，返回预测的风险标签（High Risk 或 Low Risk）以及正类概率（置信度）。
    
    要求 kwargs 中必须包含如下特征（如有缺失则使用默认值）：
      - male: 性别，1 表示男性，0 表示女性
      - age: 年龄，默认为 50
      - currentSmoker: 是否吸烟（True/False），默认为 False
      - cigsPerDay: 每天吸烟数量，默认为 0
      - BPMeds: 是否服用降压药（True/False），默认为 False
      - diabetes: 是否患糖尿病（True/False），默认为 False
      - totChol: 总胆固醇水平，默认为 200
      - sysBP: 收缩压（必须提供）
      - diaBP: 舒张压（必须提供）
      - BMI: 体质指数，默认为 25
      - heartRate: 心率，默认为 72
      - glucose: 血糖值，默认为 100
    """
    # 构造输入特征 DataFrame，特征顺序需与训练时一致
    input_data = pd.DataFrame({
        'male': [1 if kwargs.get('male', 0) == 1 else 0],
        'age': [kwargs.get('age', 50)],
        'currentSmoker': [1 if kwargs.get('currentSmoker', False) else 0],
        'cigsPerDay': [kwargs.get('cigsPerDay', 0)],
        'BPMeds': [1 if kwargs.get('BPMeds', False) else 0],
        'diabetes': [1 if kwargs.get('diabetes', False) else 0],
        'totChol': [kwargs.get('totChol', 200)],
        'sysBP': [kwargs['sysBP']],   # 必须提供
        'diaBP': [kwargs['diaBP']],   # 必须提供
        'BMI': [kwargs.get('BMI', 25)],
        'heartRate': [kwargs.get('heartRate', 72)],
        'glucose': [kwargs.get('glucose', 100)]
    })
    
    # 模型预测：假设 model.predict 返回标签（1：高风险，0：低风险）
    predicted_label = model.predict(input_data)[0]
    # model.predict_proba 返回正类概率
    confidence = model.predict_proba(input_data)[0][1]
    
    risk_status = "High Risk" if predicted_label == 1 else "Low Risk"
    
    return risk_status, float(confidence)

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('currentPassword')
    new_password = request.form.get('newPassword')
    
    # 验证当前密码
    if not current_user.verify_password(current_password):
        flash('Current password is incorrect', 'danger')
        return redirect(url_for('personal_center'))
    
    # 更新密码
    try:
        current_user.password = new_password
        db.session.commit()
        flash('Password changed successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error changing password: {str(e)}', 'danger')
    
    return redirect(url_for('personal_center'))

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    password = request.form.get('confirmPassword')
    
    # 验证密码
    if not current_user.verify_password(password):
        flash('Password is incorrect', 'danger')
        return redirect(url_for('personal_center'))
    
    # 删除账户
    try:
        # 这里应该先删除所有相关数据（如血压记录等）
        # 然后再删除用户
        
        db.session.delete(current_user)
        db.session.commit()
        logout_user()
        flash('Your account has been deleted', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting account: {str(e)}', 'danger')
        return redirect(url_for('personal_center'))
    
# 创建 OpenAI 客户端（使用 OpenAI SDK）
client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',  # 注意需要包含 /v1 后缀
    api_key='sk-CRzCVcjeuipYRrm1BbzdFN0va2RRqG6180f8G5pKJDWyqCL7'  # 请替换为你的 API Key
)

def call_qwen_api(prompt):
    """
    调用 Qwen 大模型接口，返回机器人回复文本
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",  # 如需使用其他兼容模型，请修改这里
        max_tokens=300,
        temperature=0.7,
    )
    # 假设返回结果中 choices[0].message.content 包含回复文本
    return response.choices[0].message.content

def call_qwen_model_initial(bp_data):
    """
    构造初始评估的 prompt，将用户每条记录的完整信息传递给大模型，
    并返回 Markdown 格式的回复。
    """
    prompt = (
        "You are a blood pressure assistant.\n"
        "User: My blood pressure data:\n"
    )
    for record in bp_data:
        prompt += (
            f"Data: {record['date']} Time: {record['time']}, "
            f"systolic: {record['systolic']}, diastolic: {record['diastolic']}, "
            f"status: {record['status']}\n"
        )
    prompt += (
        "\nPlease provide health advice based on the user's input, and then give advice on how to improve their health.\n"
        "Don't repeat the user's input.\n"
        "Give me a short and concise response.\n"
        "No more than 100 words.\n"
        "Assistant:"
    )
    return call_qwen_api(prompt)

def call_qwen_model_chat(user_message):
    """
    多轮对话：直接将用户消息传递给大模型并返回回复
    """
    prompt = user_message
    return call_qwen_api(prompt)

@app.route('/health_tips')
@login_required
def health_tips():
    """
    健康建议页面：
      - 页面初始加载时显示等待信息，后续通过 AJAX 获取大模型回复
    """
    print("Loading health tips page...")
    # 初始对话记录，显示等待提示
    conversation = [{"sender": "bot", "message": "正在获取初始评估，请稍候..."}]
    return render_template("health_tips.html", conversation=conversation)

@app.route('/health_tips_initial')
@login_required
def health_tips_initial():
    """
    异步接口：查询当前用户血压记录，调用大模型进行初始评估，
    返回 Markdown 格式回复。
    """
    records = BloodPressureRecord.query.filter_by(user_id=current_user.id)\
                .order_by(BloodPressureRecord.recorded_at.asc()).all()
    bp_data = []
    for record in records:
        bp_data.append({
            "date": record.recorded_at.strftime("%Y-%m-%d"),
            "time": record.recorded_at.strftime("%H:%M"),
            "systolic": record.systolic,
            "diastolic": record.diastolic,
            "status": record.status  # 例如 "正常", "偏高", "高"
        })
    initial_response = call_qwen_model_initial(bp_data)
    return jsonify({"response": initial_response})

@app.route('/health_tips_chat', methods=["POST"])
@login_required
def health_tips_chat():
    """
    多轮对话接口：接收用户消息，通过大模型 API 返回回复
    """
    user_message = request.form.get("message")
    if not user_message:
        return jsonify({"response": "No message provided"})
    bot_response = call_qwen_model_chat(user_message)
    return jsonify({"response": bot_response})

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)