import re
from collections import defaultdict


# ---------- Student Class ----------
class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def activity_summary(self):
        logins = sum(1 for a, _, _ in self.activities if a == "LOGIN")
        submissions = sum(1 for a, _, _ in self.activities if a == "SUBMIT_ASSIGNMENT")
        return logins, submissions


# ---------- Generator to read log file ----------
def activity_generator(filename):
    with open(filename, "r") as file:
        for line in file:
            try:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 5:
                    raise ValueError("Invalid format")

                student_id, name, activity, date, time = parts

                # Validate student ID
                if not re.match(r"^S\d+$", student_id):
                    raise ValueError("Invalid Student ID")

                # Validate activity
                if activity not in {"LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"}:
                    raise ValueError("Invalid Activity Type")

                # Validate date & time
                if not re.match(r"\d{4}-\d{2}-\d{2}", date):
                    raise ValueError("Invalid Date")
                if not re.match(r"\d{2}:\d{2}", time):
                    raise ValueError("Invalid Time")

                yield student_id, name, activity, date, time

            except Exception as e:
                print(
                    f"Skipping invalid entry: {line.strip()} ({e})",
                    flush=True
                )


# ---------- Main Processing ----------
students = {}
daily_stats = defaultdict(int)
abnormal_logins = defaultdict(int)

# Read file using generator
for sid, name, activity, date, time in activity_generator("student_log.txt"):
    if sid not in students:
        students[sid] = Student(sid, name)

    students[sid].add_activity(activity, date, time)
    daily_stats[date] += 1

    if activity == "LOGIN":
        abnormal_logins[sid] += 1
    elif activity == "LOGOUT":
        abnormal_logins[sid] -= 1


# ---------- Generate Report ----------
with open("activity_report.txt", "w") as output:

    print("\nSTUDENT ACTIVITY REPORT\n", flush=True)
    output.write("STUDENT ACTIVITY REPORT\n\n")

    for student in students.values():
        logins, submissions = student.activity_summary()

        report = (
            f"Student ID   : {student.student_id}\n"
            f"Name         : {student.name}\n"
            f"Total Logins : {logins}\n"
            f"Submissions  : {submissions}\n"
        )

        print(report, flush=True)
        output.write(report + "\n")

        # Detect abnormal behavior
        if abnormal_logins[student.student_id] > 0:
            warning = "⚠ Abnormal behavior: Multiple logins without logout\n"
            print(warning, flush=True)
            output.write(warning)

    # ---------- Daily Activity Statistics ----------
    print("\nDAILY ACTIVITY STATISTICS\n", flush=True)
    output.write("\nDAILY ACTIVITY STATISTICS\n\n")

    for date, count in sorted(daily_stats.items()):
        line = f"{date} -> {count} activities\n"
        print(line, flush=True)
        output.write(line)