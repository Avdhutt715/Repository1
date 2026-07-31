import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# Load Dataset
# ==========================================
dataset = pd.read_csv("data/raw/student_placement_data.csv")

# ==========================================
# Load Model & Scaler
# ==========================================
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ==========================================
# Sidebar
# ==========================================
st.sidebar.image(
    "https://img.icons8.com/color/96/student-male--v1.png",
    width=90
)

st.sidebar.title("🎓 Student Placement")

st.sidebar.success("✅ Model Loaded Successfully")
st.sidebar.info("Algorithm: Logistic Regression")
st.sidebar.metric("Accuracy", "98.50%")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Analytics",
        "🎯 Predict Placement"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("### Dataset Summary")

st.sidebar.metric(
    "Students",
    len(dataset)
)

st.sidebar.metric(
    "Placed",
    int(dataset["Placement_Status"].sum())
)

st.sidebar.metric(
    "Not Placed",
    int(len(dataset) - dataset["Placement_Status"].sum())
)

st.sidebar.metric(
    "Placement Rate",
    f"{dataset['Placement_Status'].mean()*100:.2f}%"
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🎓 Student Placement Prediction System")

    st.markdown("""
This application predicts whether a student is likely to be placed based on academic performance, technical skills, aptitude, internships, projects, certifications, attendance, and other important factors using a Machine Learning model.

### Features

- 📊 Interactive Dashboard
- 🎯 Placement Prediction
- 📈 Analytics
- 📋 Student Summary
- 📥 Download Report
""")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        len(dataset)
    )

    c2.metric(
        "Placed",
        int(dataset["Placement_Status"].sum())
    )

    c3.metric(
        "Not Placed",
        int(len(dataset) - dataset["Placement_Status"].sum())
    )

    c4.metric(
        "Placement Rate",
        f"{dataset['Placement_Status'].mean()*100:.2f}%"
    )

    st.divider()

    st.subheader("📈 Department-wise Students")

    fig = px.histogram(
        dataset,
        x="Department",
        color="Department",
        title="Students by Department"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="home_department_chart"
    )

    st.subheader("📌 Quick Statistics")

    st.dataframe(
        dataset.describe(),
        use_container_width=True
    )

    # ==========================================
# DATASET ANALYTICS PAGE
# ==========================================

elif page == "📊 Dataset Analytics":

    st.title("📊 Dataset Analytics Dashboard")

    st.sidebar.header("Analytics Filters")

    # -----------------------------
    # Filters
    # -----------------------------

    department_filter = st.sidebar.multiselect(
        "Department",
        sorted(dataset["Department"].unique()),
        default=sorted(dataset["Department"].unique())
    )

    gender_filter = st.sidebar.multiselect(
        "Gender",
        sorted(dataset["Gender"].unique()),
        default=sorted(dataset["Gender"].unique())
    )

    placement_filter = st.sidebar.multiselect(
        "Placement Status",
        [0, 1],
        default=[0, 1]
    )

    cgpa_range = st.sidebar.slider(
        "CGPA Range",
        float(dataset["CGPA"].min()),
        float(dataset["CGPA"].max()),
        (
            float(dataset["CGPA"].min()),
            float(dataset["CGPA"].max())
        )
    )

    attendance_range = st.sidebar.slider(
        "Attendance Range",
        int(dataset["Attendance"].min()),
        int(dataset["Attendance"].max()),
        (
            int(dataset["Attendance"].min()),
            int(dataset["Attendance"].max())
        )
    )

    # -----------------------------
    # Apply Filters
    # -----------------------------

    filtered_df = dataset[
        (dataset["Department"].isin(department_filter))
        &
        (dataset["Gender"].isin(gender_filter))
        &
        (dataset["Placement_Status"].isin(placement_filter))
        &
        (dataset["CGPA"].between(cgpa_range[0], cgpa_range[1]))
        &
        (dataset["Attendance"].between(attendance_range[0], attendance_range[1]))
    ]

    # -----------------------------
    # Search Student
    # -----------------------------

    search = st.text_input(
        "🔍 Search Student Name or ID"
    )

    if search:

        filtered_df = filtered_df[
            filtered_df["Name"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            filtered_df["Student_ID"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.divider()

    # -----------------------------
    # Metrics
    # -----------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Students",
        len(filtered_df)
    )

    c2.metric(
        "Placed",
        int(filtered_df["Placement_Status"].sum())
    )

    c3.metric(
        "Not Placed",
        int(len(filtered_df) - filtered_df["Placement_Status"].sum())
    )

    if len(filtered_df) > 0:
        rate = filtered_df["Placement_Status"].mean() * 100
    else:
        rate = 0

    c4.metric(
        "Placement Rate",
        f"{rate:.2f}%"
    )

    st.divider()

    # -----------------------------
    # Dataset
    # -----------------------------

    if st.checkbox("Show Dataset"):

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

    # -----------------------------
    # CSV Download
    # -----------------------------

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Dataset",
        data=csv,
        file_name="filtered_students.csv",
        mime="text/csv"
    )

    st.divider()

    # -----------------------------
    # Placement Pie Chart
    # -----------------------------

    fig = px.pie(
        filtered_df,
        names="Placement_Status",
        title="Placement Distribution",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="pie_chart"
    )

    # -----------------------------
    # CGPA Histogram
    # -----------------------------

    fig = px.histogram(
        filtered_df,
        x="CGPA",
        color="Placement_Status",
        nbins=20,
        title="CGPA Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="cgpa_histogram"
    )

    # -----------------------------
    # Department Histogram
    # -----------------------------

    fig = px.histogram(
        filtered_df,
        x="Department",
        color="Placement_Status",
        barmode="group",
        title="Department-wise Placement"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="department_histogram"
    )

    # -----------------------------
    # Coding Rating Histogram
    # -----------------------------

    fig = px.histogram(
        filtered_df,
        x="Coding_Rating",
        color="Placement_Status",
        nbins=20,
        title="Coding Rating Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="coding_histogram"
    )

    # -----------------------------
    # Box Plot
    # -----------------------------

    fig = px.box(
        filtered_df,
        x="Placement_Status",
        y="CGPA",
        color="Placement_Status",
        title="CGPA vs Placement"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="box_plot"
    )

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------

    st.subheader("🔥 Correlation Heatmap")

    corr = filtered_df.select_dtypes(include="number").corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="heatmap"
    )

# ==========================================
# PREDICT PLACEMENT PAGE
# ==========================================

elif page == "🎯 Predict Placement":

    st.title("🎯 Student Placement Prediction")

    st.write(
        "Enter student's details to predict placement probability."
    )


    col1, col2 = st.columns(2)


    # ============================
    # LEFT SIDE INPUTS
    # ============================

    with col1:

        gender_input = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )


        age = st.slider(
            "Age",
            20,
            24,
            21
        )


        department_input = st.selectbox(
            "Department",
            [
                "Artificial Intelligence",
                "Civil",
                "Computer Engineering",
                "Electronics",
                "Information Technology",
                "Mechanical"
            ]
        )


        tenth = st.number_input(
            "10th Percentage",
            40.0,
            100.0,
            75.0
        )


        twelfth = st.number_input(
            "12th Percentage",
            40.0,
            100.0,
            75.0
        )


        cgpa = st.slider(
            "CGPA",
            5.0,
            10.0,
            7.5
        )


        attendance = st.slider(
            "Attendance",
            50,
            100,
            80
        )


        backlogs = st.slider(
            "Backlogs",
            0,
            10,
            0
        )


    # ============================
    # RIGHT SIDE INPUTS
    # ============================

    with col2:


        internships = st.slider(
            "Internships",
            0,
            5,
            1
        )


        projects = st.slider(
            "Projects",
            0,
            10,
            2
        )


        certifications = st.slider(
            "Certifications",
            0,
            10,
            2
        )


        coding = st.slider(
            "Coding Rating",
            0,
            100,
            70
        )


        aptitude = st.slider(
            "Aptitude Score",
            0,
            100,
            70
        )


        communication = st.slider(
            "Communication",
            0,
            100,
            70
        )


        technical = st.slider(
            "Technical Skill",
            0,
            100,
            70
        )


        teamwork = st.slider(
            "Teamwork",
            0,
            100,
            70
        )


        leadership = st.slider(
            "Leadership",
            0,
            100,
            70
        )


        problem_solving = st.slider(
            "Problem Solving",
            0,
            100,
            70
        )


        resume = st.slider(
            "Resume Score",
            0,
            100,
            70
        )


        interview = st.slider(
            "Mock Interview",
            0,
            100,
            70
        )

    st.divider()

    # ==========================================
    # Encode Input Values
    # ==========================================

    gender = 1 if gender_input == "Male" else 0

    department_map = {
        "Artificial Intelligence": 0,
        "Civil": 1,
        "Computer Engineering": 2,
        "Electronics": 3,
        "Information Technology": 4,
        "Mechanical": 5
    }

    department = department_map[department_input]

    # ==========================================
    # Create Input DataFrame
    # ==========================================

    input_data = pd.DataFrame(
        [[
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
            resume,
            interview
        ]],
        columns=[
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
            "Mock_Interview"
        ]
    )

    # ==========================================
    # Scale Input
    # ==========================================

    scaled = pd.DataFrame(
        scaler.transform(input_data),
        columns=input_data.columns
    )

    st.divider()

    # ==========================================
    # Prediction Button
    # ==========================================

    if st.button(
        "🎯 Predict Placement",
        use_container_width=True
    ):

        prediction = model.predict(scaled)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled)[0][1]

        st.divider()

        # ======================================
        # Prediction Result
        # ======================================

        if prediction == 1:
            st.success("🎉 Student is likely to be PLACED")
        else:
            st.error("❌ Student is likely to be NOT PLACED")

        if probability is not None:
            st.metric(
                "Placement Probability",
                f"{probability * 100:.2f}%"
            )

        st.divider()

            # ==========================================
        # Prediction Dashboard
        # ==========================================

        st.subheader("📊 Prediction Dashboard")

        col1, col2 = st.columns(2)

        # -----------------------------
        # Skill Bar Chart
        # -----------------------------

        with col1:

            fig = px.bar(
                x=[
                    "Coding",
                    "Technical",
                    "Communication",
                    "Aptitude",
                    "Resume",
                    "Interview"
                ],
                y=[
                    coding,
                    technical,
                    communication,
                    aptitude,
                    resume,
                    interview
                ],
                color=[
                    coding,
                    technical,
                    communication,
                    aptitude,
                    resume,
                    interview
                ],
                title="Student Skill Analysis"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="prediction_skill_bar"
            )

        # -----------------------------
        # Radar Chart
        # -----------------------------

        with col2:

            radar = px.line_polar(
                r=[
                    coding,
                    technical,
                    communication,
                    aptitude,
                    teamwork,
                    leadership,
                    problem_solving
                ],
                theta=[
                    "Coding",
                    "Technical",
                    "Communication",
                    "Aptitude",
                    "Teamwork",
                    "Leadership",
                    "Problem Solving"
                ],
                line_close=True,
                title="Overall Skill Radar"
            )

            radar.update_traces(fill="toself")

            st.plotly_chart(
                radar,
                use_container_width=True,
                key="prediction_radar"
            )

        st.divider()

        # ==========================================
        # Student Report
        # ==========================================

        report = pd.DataFrame({

            "Feature": [

                "Gender",
                "Age",
                "Department",
                "10th %",
                "12th %",
                "CGPA",
                "Attendance",
                "Backlogs",
                "Internships",
                "Projects",
                "Certifications",
                "Coding",
                "Aptitude",
                "Communication",
                "Technical",
                "Teamwork",
                "Leadership",
                "Problem Solving",
                "Resume",
                "Interview",
                "Prediction"

            ],

            "Value": [

                gender_input,
                age,
                department_input,
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
                resume,
                interview,
                "PLACED" if prediction == 1 else "NOT PLACED"

            ]

        })

        st.dataframe(
            report,
            use_container_width=True
        )

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(

            label="📥 Download Student Report",

            data=csv,

            file_name="student_prediction_report.csv",

            mime="text/csv"

        )

        st.divider()

    elif page == "🎯 Predict Placement":
        # ==========================================
        # Model Information
        # ==========================================

        with st.expander("ℹ Model Information"):

            st.write("**Algorithm:** Logistic Regression")
            st.write(f"**Dataset Size:** {len(dataset)}")
            st.write("**Training Split:** 80%")
            st.write("**Testing Split:** 20%")
            st.write("**Target:** Placement Status")
            st.write("**Features Used:** 20")
            st.write("**Scaler:** StandardScaler") 

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
"""
<center>

### 🎓 Student Placement Prediction System

Developed using

Python • Streamlit • Scikit-Learn • Pandas • Plotly

Created by **Avdhut Sanjay Tad**

</center>
""",
unsafe_allow_html=True
)      

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
"""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

div[data-testid="metric-container"]{
    background:#f8f9fa;
    padding:15px;
    border-radius:10px;
    border:1px solid #e6e6e6;
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    border-radius:10px;
}

</style>
""",
unsafe_allow_html=True
)