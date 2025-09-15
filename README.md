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

- Python 3.9–3.11 (64-bit)
- Webcam/camera access
- Modern web browser (Chrome/Edge)
- For public hosting: HTTPS is required for camera access

### Step 1: Install Dependencies

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

macOS notes (if dlib wheel unavailable):
```bash
brew install cmake dlib
pip install -r requirements.txt
```

### Step 2: Run the Application

Development (single command):
```bash
python app.py
```

Production (recommended):
- Linux/macOS:
```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```
- Windows:
```powershell
waitress-serve --listen=0.0.0.0:5000 app:app
```

Open `http://localhost:5000`

## Usage

### First Time Setup

1. **Access the application**: Open `http://localhost:5000` in your browser
2. **Register as Teacher**: Click "Register as Teacher" and create your account
3. **Login**: Use your credentials to log in
4. **Add Students**: Go to "Add Student" to register students in the system
5. **Add Face Images**: Place student face images in the `dataset` folder (folder name must exactly match the student name)
6. Optional: run `python encode_faces.py` to pre-generate `encodings.pkl` (the app auto-generates on first run and caches it)

### Taking Attendance

1. **Login** to your teacher account
2. **Navigate** to "Take Attendance"
3. **Start Camera**: Click "Start Camera" to activate your webcam
4. **Position Students**: Have students stand in front of the camera
5. **Capture Attendance**: Click "Capture Attendance" to process face recognition
6. **Review Results**: The server recognizes faces from the captured frame
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

   - Verify dataset folder names exactly match `Student.name`
   - Ensure images are clear, face is centered, and only one face per image
   - On first run, the app builds encodings which may take time; subsequent runs are cached (`encodings.pkl`)
   - Adjust lighting or move closer to camera; try again

3. **Import errors**
4. **Port already in use**

   - Stop any process on port 5000 or change the port: `PORT=5050 python app.py`

5. **macOS crash (double free)**

   - We run Flask with reloader and threading disabled in `app.py` to avoid dlib/OpenCV crashes. Use a production server for hosting.
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

## Notes

- `start.sh` is for macOS/Linux convenience. On Windows, run the commands shown above.
- For production, serve over HTTPS to allow browser camera access on remote hosts.
