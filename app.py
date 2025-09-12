from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import csv

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)  # 8 hour session
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db = SQLAlchemy(app)

def validate_session():
    """Validate and refresh session if needed"""
    if 'teacher_id' in session and 'login_time' in session:
        try:
            login_time = datetime.fromisoformat(session['login_time'])
            if datetime.now() - login_time > timedelta(hours=8):
                # Session expired
                session.clear()
                return False
            
            # Refresh session if it's getting old (refresh every 2 hours)
            if datetime.now() - login_time > timedelta(hours=2):
                session['login_time'] = datetime.now().isoformat()
                session.modified = True
            
            return True
        except:
            session.clear()
            return False
    return False

# Database Models
class Teacher(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(20), unique=True, nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    
    def to_dict(self):
        """Convert Student object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'roll_number': self.roll_number,
            'class_name': self.class_name
        }

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id'), nullable=False)
    student = db.relationship('Student', backref='attendance_records')

# Load known face encodings
def load_known_faces():
    dataset_path = "dataset"
    known_encodings = []
    known_names = []
    
    for student_name in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_name)
        
        if not os.path.isdir(student_folder):
            continue
        
        for file in os.listdir(student_folder):
            if file.endswith((".jpg", ".png")):
                img_path = os.path.join(student_folder, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_img)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(student_name)
                    break  # Only need one image per student
    
    return known_encodings, known_names

@app.route('/')
def index():
    if 'teacher_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        teacher = Teacher.query.filter_by(username=username).first()
        
        if teacher and check_password_hash(teacher.password_hash, password):
            # Clear any existing session and create new one
            session.clear()
            session['teacher_id'] = teacher.id
            session['teacher_name'] = teacher.name
            session['login_time'] = datetime.now().isoformat()
            session.modified = True
            
            # Debug information
            print(f"DEBUG: Teacher {teacher.name} (ID: {teacher.id}) logged in successfully")
            print(f"DEBUG: Session created with teacher_id: {session.get('teacher_id')}")
            
            flash(f'Login successful! Welcome, {teacher.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'teacher_name' in session:
        teacher_name = session['teacher_name']
        session.clear()
        flash(f'Logged out successfully, {teacher_name}!', 'success')
    else:
        session.clear()
        flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/debug_session')
def debug_session():
    """Debug route to check current session (remove in production)"""
    if 'teacher_id' in session:
        return jsonify({
            'teacher_id': session.get('teacher_id'),
            'teacher_name': session.get('teacher_name'),
            'login_time': session.get('login_time'),
            'session_valid': validate_session()
        })
    else:
        return jsonify({'message': 'No active session'})

@app.route('/force_logout')
def force_logout():
    """Force logout and clear all sessions (for testing)"""
    session.clear()
    flash('All sessions cleared. Please login again.', 'info')
    return redirect(url_for('login'))

@app.route('/list_teachers')
def list_teachers():
    """List all teachers (for debugging - remove in production)"""
    teachers = Teacher.query.all()
    teacher_list = []
    for teacher in teachers:
        teacher_list.append({
            'id': teacher.id,
            'username': teacher.username,
            'name': teacher.name,
            'email': teacher.email
        })
    return jsonify(teacher_list)

@app.route('/dashboard')
def dashboard():
    if not validate_session():
        return redirect(url_for('login'))
    
    # Get today's attendance
    today = datetime.now().date()
    today_attendance = Attendance.query.filter_by(date=today).all()
    
    # Get all students and convert to dictionaries for JSON serialization
    students = Student.query.all()
    students_data = [student.to_dict() for student in students]
    
    return render_template('dashboard.html', 
                         students=students_data, 
                         today_attendance=today_attendance,
                         today=today)

@app.route('/take_attendance')
def take_attendance():
    if not validate_session():
        return redirect(url_for('login'))
    
    students = Student.query.all()
    # Convert Student objects to dictionaries for JSON serialization
    students_data = [student.to_dict() for student in students]
    
    return render_template('take_attendance.html', students=students_data)

@app.route('/api/process_attendance', methods=['POST'])
def process_attendance():
    if not validate_session():
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Load known faces
        known_encodings, known_names = load_known_faces()
        
        # Get camera frame (this would need to be implemented with proper camera access)
        # For now, we'll simulate the process
        
        # Mark attendance for detected students
        detected_students = request.json.get('detected_students', [])
        today = datetime.now().date()
        current_time = datetime.now().time()
        
        marked_count = 0
        for student_data in detected_students:
            # Handle both string names and dictionary objects
            if isinstance(student_data, dict):
                student_name = student_data.get('name')
            else:
                student_name = student_data
                
            student = Student.query.filter_by(name=student_name).first()
            if student:
                # Check if attendance already marked
                existing = Attendance.query.filter_by(
                    student_id=student.id, 
                    date=today
                ).first()
                
                if not existing:
                    attendance = Attendance(
                        student_id=student.id,
                        date=today,
                        time=current_time,
                        teacher_id=session['teacher_id']
                    )
                    db.session.add(attendance)
                    marked_count += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Attendance marked for {marked_count} students',
            'marked_count': marked_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_attendance')
def export_attendance():
    if not validate_session():
        return redirect(url_for('login'))
    
    # Get today's attendance
    today = datetime.now().date()
    today_attendance = Attendance.query.filter_by(date=today).all()
    
    # Get all students to show absent ones
    all_students = Student.query.all()
    present_student_ids = {att.student_id for att in today_attendance}
    
    # Create CSV content
    csv_content = "Student Name,Roll Number,Class,Date,Time,Status\n"
    
    # Add present students
    for attendance in today_attendance:
        student = attendance.student
        csv_content += f'"{student.name}","{student.roll_number}","{student.class_name}","{today.strftime("%Y-%m-%d")}","{attendance.time.strftime("%H:%M:%S")}","Present"\n'
    
    # Add absent students
    for student in all_students:
        if student.id not in present_student_ids:
            csv_content += f'"{student.name}","{student.roll_number}","{student.class_name}","{today.strftime("%Y-%m-%d")}","","Absent"\n'
    
    # Create response with CSV file
    from flask import Response
    response = Response(csv_content, mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename=attendance_{today.strftime("%Y-%m-%d")}.csv'
    
    return response

@app.route('/export_attendance_range')
def export_attendance_range():
    if not validate_session():
        return redirect(url_for('login'))
    
    # Get date range from query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({'error': 'Start date and end date are required'}), 400
    
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # Get attendance for date range
    attendance_records = Attendance.query.filter(
        Attendance.date >= start_date,
        Attendance.date <= end_date
    ).order_by(Attendance.date, Attendance.time).all()
    
    # Get all students
    all_students = Student.query.all()
    
    # Create CSV content
    csv_content = "Student Name,Roll Number,Class,Date,Time,Status\n"
    
    # Group attendance by date
    attendance_by_date = {}
    for att in attendance_records:
        date_str = att.date.strftime('%Y-%m-%d')
        if date_str not in attendance_by_date:
            attendance_by_date[date_str] = []
        attendance_by_date[date_str].append(att)
    
    # Generate CSV for each date in range
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        present_student_ids = {att.student_id for att in attendance_by_date.get(date_str, [])}
        
        # Add present students for this date
        for attendance in attendance_by_date.get(date_str, []):
            student = attendance.student
            csv_content += f'"{student.name}","{student.roll_number}","{student.class_name}","{date_str}","{attendance.time.strftime("%H:%M:%S")}","Present"\n'
        
        # Add absent students for this date
        for student in all_students:
            if student.id not in present_student_ids:
                csv_content += f'"{student.name}","{student.roll_number}","{student.class_name}","{date_str}","","Absent"\n'
        
        current_date = current_date + timedelta(days=1)
    
    # Create response with CSV file
    from flask import Response
    filename = f'attendance_{start_date.strftime("%Y-%m-%d")}_to_{end_date.strftime("%Y-%m-%d")}.csv'
    response = Response(csv_content, mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    
    return response

@app.route('/register_teacher', methods=['GET', 'POST'])
def register_teacher():
    # Clear any existing session when registering
    session.clear()
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']
        
        # Check if username already exists
        if Teacher.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register_teacher.html')
        
        # Check if email already exists
        if Teacher.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return render_template('register_teacher.html')
        
        # Create new teacher
        teacher = Teacher(
            username=username,
            password_hash=generate_password_hash(password),
            name=name,
            email=email
        )
        
        db.session.add(teacher)
        db.session.commit()
        
        # Verify the teacher was created
        created_teacher = Teacher.query.filter_by(username=username).first()
        if created_teacher:
            flash(f'Teacher "{name}" registered successfully with username "{username}"! Please login.', 'success')
        else:
            flash('Registration completed but verification failed. Please try logging in.', 'warning')
        
        return redirect(url_for('login'))
    
    return render_template('register_teacher.html')

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if not validate_session():
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        class_name = request.form['class_name']
        
        # Check if roll number already exists
        if Student.query.filter_by(roll_number=roll_number).first():
            flash('Roll number already exists', 'error')
            return render_template('add_student.html')
        
        # Create new student
        student = Student(
            name=name,
            roll_number=roll_number,
            class_name=class_name
        )
        
        db.session.add(student)
        db.session.commit()
        
        flash('Student added successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('add_student.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create a default teacher if none exists
        if not Teacher.query.first():
            default_teacher = Teacher(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                name='Administrator',
                email='admin@school.com'
            )
            db.session.add(default_teacher)
            db.session.commit()
            print("Default teacher created: username='admin', password='admin123'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
