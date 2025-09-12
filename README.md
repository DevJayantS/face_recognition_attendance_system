# AI Attendance System

A modern web application for automated student attendance tracking using face recognition technology. Built with Flask, OpenCV, and face_recognition library.

## Features

- 🎯 **Face Recognition**: Advanced AI-powered student identification
- 👨‍🏫 **Teacher Authentication**: Secure login system for teachers
- 📊 **Real-time Dashboard**: Live attendance statistics and reports
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 📷 **Camera Integration**: Webcam-based attendance capture
- 💾 **Database Storage**: SQLite database for attendance records
- 📈 **Attendance Reports**: Comprehensive tracking and analytics

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/camera access
- Modern web browser

### Step 1: Install Dependencies

```bash
pip3 install -r requirements.txt
```

**Note**: If you encounter issues with `face_recognition` on macOS, you may need to install additional dependencies:

```bash
# For macOS
brew install cmake
brew install dlib
pip3 install face_recognition
```

### Step 2: Run the Application

```bash
python3 app.py
```

The application will be available at: `http://localhost:5000`

## Usage

### First Time Setup

1. **Access the application**: Open `http://localhost:5000` in your browser
2. **Register as Teacher**: Click "Register as Teacher" and create your account
3. **Login**: Use your credentials to log in
4. **Add Students**: Go to "Add Student" to register students in the system
5. **Add Face Images**: Place student face images in the `dataset` folder

### Taking Attendance

1. **Login** to your teacher account
2. **Navigate** to "Take Attendance"
3. **Start Camera**: Click "Start Camera" to activate your webcam
4. **Position Students**: Have students stand in front of the camera
5. **Capture Attendance**: Click "Capture Attendance" to process face recognition
6. **Review Results**: Check the detected students in the modal
7. **Save**: Click "Save to Database" to record attendance

## Default Credentials

When you first run the application, a default teacher account is created:

- **Username**: `admin`
- **Password**: `admin123`

**Important**: Change these credentials after first login for security!

## File Structure

```
SIH - AI attendance/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
├── static/               # Static assets (CSS, JS)
├── dataset/              # Student face images
└── ...                   # Other files
```

## Troubleshooting

### Common Issues

1. **Camera not working**

   - Ensure camera permissions are granted
   - Check if camera is being used by another application

2. **Face recognition not working**

   - Verify student images are in the correct dataset folder
   - Ensure images are clear and show faces prominently

3. **Import errors**
   - Verify all requirements are installed: `pip3 install -r requirements.txt`
   - Check Python version compatibility

## Security Considerations

- Change default admin credentials immediately
- Use HTTPS in production environments
- Regularly backup the database
- Implement proper session management

## Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the error logs in the console
3. Ensure all dependencies are properly installed
4. Verify your dataset structure is correct
