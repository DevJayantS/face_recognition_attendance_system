#!/usr/bin/env bash

echo "🚀 Starting AI Attendance System..."
echo "=================================="

# Check if Python 3 is installed
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import flask, cv2, face_recognition, openpyxl" >/dev/null 2>&1; then
    echo "⚠️  Some dependencies are missing. Installing requirements..."
    pip3 install -r requirements.txt
fi

# Check if database exists, if not run setup
if [ ! -f "attendance.db" ]; then
    echo "🗄️  Setting up database..."
    python3 setup_database.py
fi

echo "🌐 Starting web application..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Start the Flask application
python3 app.py
