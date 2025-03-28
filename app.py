from flask import Flask, render_template, request
from flask import jsonify
from flask import redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import re
from flask_migrate import Migrate
from enum import Enum

from datetime import datetime

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

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
def data_analysis():
    # 示例数据
    # 后端数据库接入后修改
    records = [
        {"date": "2025/3/10", "time": "23:50", "systolic": 120, "diastolic": 80, "status": "正常"},
        {"date": "2025/3/11", "time": "01:04", "systolic": 130, "diastolic": 90, "status": "偏高"},
        {"date": "2025/3/12", "time": "22:30", "systolic": 125, "diastolic": 85, "status": "正常"},
        {"date": "2025/3/13", "time": "23:15", "systolic": 135, "diastolic": 95, "status": "高"},
        {"date": "2025/3/14", "time": "00:45", "systolic": 128, "diastolic": 88, "status": "偏高"},
    ]
    return render_template('data_analysis.html', records=records)

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
        risk_score = calculate_risk_score(
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

        if sysBP >= 140 or diaBP >= 90:
            result = {
                'status': 'warning',
                'message': 'Your blood pressure is high. Please consult a doctor.',
                'details': {
                    'risk_score': risk_score,
                    'recommendations': [
                        'Regular blood pressure monitoring',
                        'Reduce sodium intake',
                        'Increase physical activity',
                        'Consult with a healthcare provider'
                    ]
                }
            }
        else:
            result = {
                'status': 'success',
                'message': 'Your blood pressure is normal. Keep maintaining a healthy lifestyle.',
                'details': {
                    'risk_score': risk_score,
                    'recommendations': [
                        'Continue healthy diet',
                        'Regular exercise',
                        'Annual check-ups'
                    ]
                }
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500

def calculate_risk_score(**kwargs):
    """Example risk score calculation - replace with your actual algorithm"""
    risk = 0
    
    # Simple example calculation (replace with your actual algorithm)
    if kwargs['sysBP'] >= 140: risk += 2
    if kwargs['diaBP'] >= 90: risk += 2
    if kwargs['currentSmoker']: risk += 1
    if kwargs['diabetes']: risk += 1
    if kwargs['BPMeds']: risk += 1
    if kwargs['age'] > 50: risk += 1
    if kwargs['BMI'] > 30: risk += 1
    
    return f"{risk}/8 risk factors"  # Example format

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

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)