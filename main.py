import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if not exists:
        mess.showerror(title='Some file missing', message='Please contact us for help')
        window.destroy()

def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        with open("TrainingImageLabel/psd.txt", "r") as tf:
            key = tf.read()
    else:
        new_pas = "CAIT"
        with open("TrainingImageLabel/psd.txt", "w") as tf:
            tf.write(new_pas)
        mess.showinfo(title='Password Registered', message='New password was registered successfully!!')
        return

    password = tsd.askstring('Password', 'Enter Password', show='*')
    if (password == key):
        TrainImages()
    elif (password is None):
        pass
    else:
        mess.showerror(title='Wrong Password', message='You have entered the wrong password')

def show_students():
    global student_window
    student_window = tk.Toplevel(window)
    student_window.title("Student Details")
    student_window.geometry("400x300")
    student_window.configure(background="#f0f0f0")

    tree = ttk.Treeview(student_window, columns=("ID", "Name"), show='headings')
    tree.heading("ID", text="ID")
    tree.heading("Name", text="Name")
    tree.pack(fill='both', expand=True)

    if os.path.isfile('Student_Details/StudentDetails.csv'):
        with open('Student_Details/StudentDetails.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 2:
                    tree.insert('', 'end', values=row)

def verify_password_for_registration():
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password == "CAIT":
        show_register_window()
    else:
        mess.showerror(title='Wrong Password', message='You have entered the wrong password')

def show_register_window():
    global register_window, id_entry, name_entry
    register_window = tk.Toplevel(window)
    register_window.title("Register New Student")
    register_window.geometry("300x200")
    register_window.configure(background="#f0f0f0")

    tk.Label(register_window, text='Enter ID:', font=('Arial', 12, 'bold'), bg="#f0f0f0").pack(pady=10)
    id_entry = tk.Entry(register_window, font=('Arial', 12), width=20)
    id_entry.pack(pady=5)

    tk.Label(register_window, text='Enter Name:', font=('Arial', 12, 'bold'), bg="#f0f0f0").pack(pady=10)
    name_entry = tk.Entry(register_window, font=('Arial', 12), width=20)
    name_entry.pack(pady=5)

    tk.Button(register_window, text="Save Profile", command=save_profile, font=('Arial', 12, 'bold'), bg="#4CAF50", fg="white").pack(pady=10)
    tk.Button(register_window, text="Take Image", command=lambda: take_image(id_entry.get(), name_entry.get()), font=('Arial', 12, 'bold'), bg="#FF5722", fg="white").pack(pady=10)

def save_profile():
    Id = id_entry.get()
    name = name_entry.get()

    if name.isalpha() or ' ' in name:
        register_window.destroy()
        take_image(Id, name)
    else:
        if not name.isalpha():
            mess.showerror(title='Error', message='Enter Correct name')

def take_image(Id, name):
    assure_path_exists("TrainingImage/")
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite("TrainingImage/" + name + "." + str(sampleNum) + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('Registering New Student', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 100:
            break
    cam.release()
    cv2.destroyAllWindows()

    res = "Images Taken for ID : " + Id
    row = [Id, name]
    with open('Student_Details/StudentDetails.csv', 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    mess.showinfo(title='Success', message=res)

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    faces, Ids = getImagesAndLabels("TrainingImage/")
    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    mess.showinfo(title='Success', message="Training dataset generated and saved")
    print("Training dataset generated and saved")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')
        imgNp = np.array(img, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[2])
        faces.append(imgNp)
        Ids.append(Id)
    return faces, Ids

def get_student_name_by_id(student_id):
    """Get student name by ID from the StudentDetails.csv file."""
    if os.path.isfile('Student_Details/StudentDetails.csv'):
        with open('Student_Details/StudentDetails.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 2 and row[0] == student_id:
                    return row[1]
    return "Unknown"

def create_daily_attendance_file():
    """Create a new CSV file for today’s date."""
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    attendance_file = f"Attendance/Attendance_{date_str}.csv"
    column = ['ID', 'NAME', 'TIME']
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(column)
    return attendance_file

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance_file = create_daily_attendance_file()
    logged_students = set()
    
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            name = get_student_name_by_id(str(id))
            if confidence < 100 and name != "Unknown":
                id_display = f"ID : {id}"
                if id not in logged_students:
                    logged_students.add(id)
                    timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    row = [id_display, name, timeStamp]
                    with open(attendance_file, 'a+', newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(row)
            else:
                id_display = "Unknown"
                name = "Unknown"
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, str(id_display), (x, y + h), font, 1, (255, 0, 0), 2)
        cv2.imshow('Tracking Images', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def TakeAttendance():
    TrackImages()

def main():
    global clock, window
    window = tk.Tk()
    window.title("Face Recognition Attendance System")
    window.geometry("800x600")
    window.configure(background="#f0f0f0")

    tk.Label(window, text='Face Recognition Attendance System', font=('Arial', 16, 'bold'), bg="#f0f0f0", fg="#333333").pack(pady=20)

    button_frame = tk.Frame(window, bg="#f0f0f0")
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Register New Student", command=verify_password_for_registration, font=('Arial', 12, 'bold'), bg="#4CAF50", fg="white", relief='flat', padx=20, pady=10).grid(row=0, column=0, padx=10, pady=10)
    tk.Button(button_frame, text="Update System", command=TrainImages, font=('Arial', 12, 'bold'), bg="#2196F3", fg="white", relief='flat', padx=20, pady=10).grid(row=0, column=1, padx=10, pady=10)
    tk.Button(button_frame, text="Take Attendance", command=TakeAttendance, font=('Arial', 12, 'bold'), bg="#FF5722", fg="white", relief='flat', padx=20, pady=10).grid(row=0, column=2, padx=10, pady=10)
    tk.Button(button_frame, text="Show Students", command=show_students, font=('Arial', 12, 'bold'), bg="#9E9E9E", fg="white", relief='flat', padx=20, pady=10).grid(row=0, column=3, padx=10, pady=10)

    clock = tk.Label(window, font=('Arial', 12, 'bold'), bg="#f0f0f0", fg="#333333")
    clock.pack(side='bottom', pady=20)

    tick()
    window.mainloop()

if __name__ == "__main__":
    main()
