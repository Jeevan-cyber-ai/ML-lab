class Student:
    def __init__(self, name, assignment, internal, attendance, study_hours):
        self.name = name
        self.assignment = assignment
        self.internal = internal
        self.attendance = attendance
        self.study_hours = study_hours

    # Function to calculate average score
    def calculate_average(self):
        return (self.assignment + self.internal) / 2

    # Function to determine academic risk level
    def risk_level(self):
        avg = self.calculate_average()

        if avg >= 75 and self.attendance >= 80:
            return "Low Risk"
        elif avg >= 50 and self.attendance >= 65:
            return "Moderate Risk"
        else:
            return "High Risk"

    # Display structured performance report
    def display_report(self):
        print("-" * 40)
        print(f"Student Name        : {self.name}")
        print(f"Assignment Score    : {self.assignment}")
        print(f"Internal Test Score : {self.internal}")
        print(f"Attendance (%)      : {self.attendance}")
        print(f"Study Hours / Week  : {self.study_hours}")
        print(f"Average Score       : {self.calculate_average():.2f}")
        print(f"Academic Risk Level : {self.risk_level()}")
        print("-" * 40)

n = int(input("Enter number of students: "))

students = []
# Store student data in a file
with open("student_data.txt", "w") as file:
# Use loop to process multiple student records
    for i in range(n):
        print(f"\nEnter details of student {i+1}:")
        name = input("Name: ")
        assignment = float(input("Assignment score: "))
        internal = float(input("Internal test score: "))
        attendance = float(input("Attendance percentage: "))
        study_hours = float(input("Hours studied per week: "))

        student = Student(name, assignment, internal, attendance, study_hours)
        students.append(student)

        # Write data to file
        file.write(f"{name},{assignment},{internal},{attendance},{study_hours}\n")



# Display performance reports
print("\nSTUDENT PERFORMANCE REPORT")
for student in students:
    student.display_report()