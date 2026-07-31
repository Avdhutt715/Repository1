import pandas as pd
from faker import Faker
import random
import os

fake = Faker("en_IN")

NUM_STUDENTS = 1000

departments = [
    "Computer Engineering",
    "Information Technology",
    "Artificial Intelligence",
    "Electronics",
    "Mechanical",
    "Civil"
]

genders = ["Male", "Female"]

students = []

for i in range(1, NUM_STUDENTS + 1):

    student_id = f"STU{i:04d}"
    name = fake.name()
    gender = random.choice(genders)
    age = random.randint(20, 24)
    department = random.choice(departments)

    # Academic Details
    tenth = round(random.uniform(60, 98), 2)
    twelfth = round(random.uniform(60, 98), 2)
    cgpa = round(random.uniform(5.5, 9.9), 2)

    # College Performance
    attendance = random.randint(60, 100)
    backlogs = random.randint(0, 5)

    # Skills
    coding = random.randint(30, 100)
    communication = random.randint(40, 100)
    technical = random.randint(35, 100)
    aptitude = random.randint(30, 100)

    # Experience
    internships = random.randint(0, 3)
    projects = random.randint(1, 6)
    certifications = random.randint(0, 8)

    # Soft Skills
    teamwork = random.randint(50, 100)
    leadership = random.randint(40, 100)
    problem_solving = random.randint(40, 100)

    # Resume
    resume_score = random.randint(40, 100)
    mock_interview = random.randint(30, 100)

    # Placement Score
    placement_score = 0

    placement_score += (cgpa / 10) * 20
    placement_score += (tenth / 100) * 10
    placement_score += (twelfth / 100) * 10
    placement_score += (coding / 100) * 15
    placement_score += (technical / 100) * 10
    placement_score += (communication / 100) * 10
    placement_score += (aptitude / 100) * 10

    placement_score += internships * 3
    placement_score += projects * 2
    placement_score += certifications

    placement_score += (attendance / 100) * 5

    placement_score -= backlogs * 3

    if internships >= 2:
        placement_score += 3

    if projects >= 4:
        placement_score += 2

    if cgpa >= 8.5:
        placement_score += 3

    placement_score = round(placement_score, 2)

    placement_status = 1 if placement_score >= 75 else 0

    students.append([
        student_id,
        name,
        gender,
        age,
        department,
        tenth,
        twelfth,
        cgpa,
        attendance,
        backlogs,
        internships,
        projects,
        certifications,
        coding,
        aptitude,
        communication,
        technical,
        teamwork,
        leadership,
        problem_solving,
        resume_score,
        mock_interview,
        placement_score,
        placement_status
    ])

columns = [
    "Student_ID",
    "Name",
    "Gender",
    "Age",
    "Department",
    "10th_Percentage",
    "12th_Percentage",
    "CGPA",
    "Attendance",
    "Backlogs",
    "Internships",
    "Projects",
    "Certifications",
    "Coding_Rating",
    "Aptitude_Score",
    "Communication",
    "Technical_Skill",
    "Teamwork",
    "Leadership",
    "Problem_Solving",
    "Resume_Score",
    "Mock_Interview",
    "Placement_Score",
    "Placement_Status"
]

df = pd.DataFrame(students, columns=columns)

os.makedirs("data/raw", exist_ok=True)

df.to_csv("data/raw/student_placement_data.csv", index=False)

print(df.head())
print("\nTotal Students:", len(df))
print("\nPlacement Distribution:")
print(df["Placement_Status"].value_counts())