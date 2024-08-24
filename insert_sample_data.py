import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

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

# Create sample data
def insert_sample_data():
    with app.app_context():
        # Check if there are existing records
        existing_records = Attendance.query.count()
        if existing_records > 0:
            print("Sample data already exists.")
            return

        # Sample data
        sample_data = [
            Attendance(employee_id=1, date=datetime(2024, 8, 1).date(), checkin_time='09:00', checkout_time='17:00', status='Present'),
            Attendance(employee_id=2, date=datetime(2024, 8, 1).date(), checkin_time='09:15', checkout_time='17:00', status='Present'),
            Attendance(employee_id=1, date=datetime(2024, 8, 2).date(), checkin_time='09:05', checkout_time='17:00', status='Present'),
            Attendance(employee_id=2, date=datetime(2024, 8, 2).date(), checkin_time='09:20', checkout_time='17:00', status='Present'),
        ]

        # Insert sample data
        db.session.bulk_save_objects(sample_data)
        db.session.commit()
        print("Sample data inserted successfully.")


def insert_sample_data():
    with app.app_context():
        # Sample data
        sample_data = [
            Attendance(employee_id=1, date=datetime(2024, 8, 3).date(), checkin_time='09:10', checkout_time='17:00', status='Present'),
            Attendance(employee_id=2, date=datetime(2024, 8, 3).date(), checkin_time='09:25', checkout_time='17:00', status='Present'),
        ]

        # Insert sample data
        db.session.bulk_save_objects(sample_data)
        db.session.commit()
        print("Sample data inserted successfully.")

if __name__ == "__main__":
    insert_sample_data()
