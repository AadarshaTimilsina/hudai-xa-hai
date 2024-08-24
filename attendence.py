import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask application
app = Flask(__name__)

# Configuration for SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/DELL/PycharmProjects/project3/instance/attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define the Attendance model
class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    checkin_time = db.Column(db.String(10), nullable=True)
    checkout_time = db.Column(db.String(10), nullable=True)
    status = db.Column(db.String(10), nullable=False)
    late_entry = db.Column(db.String(10), nullable=True)
    early_exit = db.Column(db.String(10), nullable=True)

# Create the database and the Attendance table
with app.app_context():
    db.create_all()
    print("Attendance database and table created successfully.")

if __name__ == "__main__":
    app.run()
