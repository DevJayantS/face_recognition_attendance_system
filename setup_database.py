#!/usr/bin/env python3
"""
Database setup script for AI Attendance System
This script will populate the database with students based on your existing dataset folder
"""

import os
import sys
from app import app, db, Student, Teacher
from werkzeug.security import generate_password_hash

def setup_database():
    """Set up the database and populate with initial data"""
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")
        
        # Check if default teacher exists
        if not Teacher.query.filter_by(username='admin').first():
            default_teacher = Teacher(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                name='Administrator',
                email='admin@school.com'
            )
            db.session.add(default_teacher)
            print("✅ Default teacher account created")
        else:
            print("ℹ️  Default teacher account already exists")
        
        # Add students based on dataset folder
        dataset_path = "dataset"
        if os.path.exists(dataset_path):
            for student_name in os.listdir(dataset_path):
                student_folder = os.path.join(dataset_path, student_name)
                
                if not os.path.isdir(student_folder):
                    continue
                
                # Check if student already exists
                existing_student = Student.query.filter_by(name=student_name).first()
                if not existing_student:
                    # Create new student
                    student = Student(
                        name=student_name,
                        roll_number=f"R{len(Student.query.all()) + 1:03d}",  # Generate roll number
                        class_name="Class 10"  # Default class, can be changed later
                    )
                    db.session.add(student)
                    print(f"✅ Added student: {student_name} (Roll: {student.roll_number})")
                else:
                    print(f"ℹ️  Student already exists: {student_name}")
        else:
            print("⚠️  Dataset folder not found. Please create it and add student images.")
        
        # Commit all changes
        try:
            db.session.commit()
            print("✅ Database setup completed successfully!")
        except Exception as e:
            print(f"❌ Error committing to database: {e}")
            db.session.rollback()
            return False
        
        return True

def show_credentials():
    """Display login credentials"""
    print("\n" + "="*50)
    print("🔐 LOGIN CREDENTIALS")
    print("="*50)
    print("Username: admin")
    print("Password: admin123")
    print("="*50)
    print("⚠️  IMPORTANT: Change these credentials after first login!")
    print("="*50)

def main():
    """Main function"""
    print("🚀 Setting up AI Attendance System Database...")
    print("-" * 50)
    
    try:
        if setup_database():
            show_credentials()
            print("\n🎉 Setup complete! You can now run the application:")
            print("   python3 app.py")
            print("\n🌐 Open your browser and go to: http://localhost:5000")
        else:
            print("\n❌ Setup failed. Please check the error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        print("Please check your Python environment and dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
