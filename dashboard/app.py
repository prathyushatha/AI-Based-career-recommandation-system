import os
import re
import joblib
import pandas as pd
import streamlit as st

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide"
)


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_career_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf.pkl"
)

TFIDF_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_matrix.pkl"
)


# =====================================================
# LOAD DATASET
# =====================================================

try:

    df = pd.read_csv(
        DATA_PATH
    )

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()


# =====================================================
# CHECK REQUIRED COLUMNS
# =====================================================

required_columns = [
    "Career",
    "Skills"
]

for column in required_columns:

    if column not in df.columns:

        st.error(
            f"Required column '{column}' "
            "is missing from the dataset."
        )

        st.stop()


# =====================================================
# LOAD TF-IDF MODEL
# =====================================================

try:

    tfidf = joblib.load(
        TFIDF_PATH
    )

    tfidf_matrix = joblib.load(
        TFIDF_MATRIX_PATH
    )

except Exception as e:

    st.error(
        f"Unable to load TF-IDF model: {e}"
    )

    st.stop()


# =====================================================
# BUILD CAREER REQUIREMENTS
# =====================================================

career_requirements = {

    "Machine Learning Engineer": [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Pandas",
        "NumPy",
        "Scikit-Learn",
        "Data Preprocessing",
        "Model Evaluation"
    ],

    "Data Scientist": [
        "Python",
        "SQL",
        "Statistics",
        "Machine Learning",
        "Pandas",
        "NumPy",
        "Data Visualization",
        "Scikit-Learn"
    ],

    "Data Analyst": [
        "Python",
        "SQL",
        "Excel",
        "Statistics",
        "Pandas",
        "Power BI",
        "Data Visualization"
    ],

    "AI Researcher": [
        "Python",
        "Mathematics",
        "Statistics",
        "Machine Learning",
        "Deep Learning",
        "Research Skills",
        "PyTorch",
        "TensorFlow"
    ],

    "Deep Learning Engineer": [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "Neural Networks",
        "CNN",
        "RNN"
    ],

    "Frontend Developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Responsive Design",
        "Git"
    ],

    "Backend Developer": [
        "Python",
        "Java",
        "Node.js",
        "SQL",
        "REST API",
        "Databases",
        "Git"
    ],

    "Software Engineer": [
        "Python",
        "Java",
        "C++",
        "Data Structures",
        "Algorithms",
        "OOP",
        "Git"
    ]

}


# =====================================================
# AVAILABLE CAREERS
# =====================================================

career_list = sorted(

    df["Career"]
    .dropna()
    .unique()
    .tolist()

)


# =====================================================
# LEARNING RESOURCES
# =====================================================

learning_resources = {

    "Python":
        "Python Basics → OOP → NumPy → Pandas → Projects",

    "Sql":
        "SQL Basics → Joins → Subqueries → Window Functions → Optimization",

    "Machine Learning":
        "Python → NumPy → Pandas → Scikit-Learn → ML Algorithms → Projects",

    "Deep Learning":
        "Python → Neural Networks → TensorFlow → PyTorch → CNN → RNN",

    "Data Analysis":
        "Excel → SQL → Pandas → Data Cleaning → Power BI → Projects",

    "Data Visualization":
        "Matplotlib → Seaborn → Plotly → Dashboard Development",

    "Java":
        "Core Java → OOP → Collections → Spring Boot",

    "C++":
        "C++ Basics → OOP → STL → DSA",

    "Html/Css":
        "HTML5 → CSS3 → Flexbox → Grid → Responsive Design",

    "Html":
        "HTML5 → Forms → Semantic HTML",

    "Css":
        "CSS3 → Flexbox → Grid → Responsive Design",

    "Javascript":
        "JavaScript → DOM → ES6 → APIs → React",

    "React":
        "React Components → Props → State → Hooks → Projects",

    "Node.Js":
        "Node.js → Express.js → REST API → MongoDB",

    "Tensorflow":
        "TensorFlow Basics → Neural Networks → CNN → Advanced Projects",

    "Pytorch":
        "PyTorch Basics → Neural Networks → Deep Learning Projects",

    "Power Bi":
        "Power BI → Data Cleaning → DAX → Dashboard Development",

    "Tableau":
        "Tableau → Charts → Dashboards → Data Storytelling",

    "Excel":
        "Excel Functions → Pivot Tables → Charts → Advanced Excel",

    "Sql Server":
        "SQL Basics → Database Design → Queries → Optimization"

}


# =====================================================
# CAREER DESCRIPTIONS
# =====================================================

career_description = {

    "Machine Learning Engineer":
        "Builds intelligent AI systems using Machine Learning algorithms.",

    "AI Researcher":
        "Researches Artificial Intelligence algorithms and develops innovative AI solutions.",

    "Deep Learning Engineer":
        "Develops Deep Learning models using TensorFlow and PyTorch.",

    "Data Scientist":
        "Uses statistics, Python, Machine Learning and data analysis to solve real-world problems.",

    "Data Analyst":
        "Analyzes business data using SQL, Excel, Python, Power BI and visualization tools.",

    "Business Analyst":
        "Analyzes business requirements and transforms data into actionable insights for decision-making.",

    "Business Manager":
        "Manages business operations, teams, planning and strategic decision-making.",

    "Software Engineer":
        "Designs, develops, tests and maintains software applications.",

    "Cloud Engineer":
        "Designs and manages cloud infrastructure using platforms such as AWS, Azure and Google Cloud.",

    "DevOps Engineer":
        "Automates software development and deployment using CI/CD, Docker and cloud technologies.",

    "Frontend Developer":
        "Builds responsive and interactive user interfaces using HTML, CSS and JavaScript.",

    "Backend Developer":
        "Develops server-side applications, APIs and database systems.",

    "Full Stack Developer":
        "Develops both frontend and backend components of modern web applications.",

    "Ethical Hacker":
        "Identifies security vulnerabilities and helps organizations protect their systems.",

    "Cyber Security Analyst":
        "Monitors systems, detects security threats and protects organizational data.",

    "Graphic Designer":
        "Creates visual content and designs using creative and digital design tools."

}
# =====================================================
# PART 2 — RECOMMENDATION + SKILL GAP + ROADMAP + CHATBOT
# =====================================================


# =====================================================
# TF-IDF CAREER RECOMMENDATION
# =====================================================

def recommend_from_input(
    skills,
    interest
):

    # Combine user skills and interest
    user_profile = (
        str(skills).lower().strip()
        + " "
        + str(interest).lower().strip()
    )

    # Convert user input into TF-IDF vector
    user_vector = tfidf.transform(
        [user_profile]
    )

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    )[0]

    # Create index-score pairs
    indexed_scores = list(
        enumerate(
            similarity_scores
        )
    )

    # Sort by highest similarity
    indexed_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    used_careers = set()

    # Select top 3 unique careers
    for index, score in indexed_scores:

        career = str(
            df.iloc[index]["Career"]
        ).strip()

        if career not in used_careers:

            recommendations.append(
                (
                    career,
                    round(
                        float(score) * 100,
                        2
                    )
                )
            )

            used_careers.add(
                career
            )

        if len(recommendations) >= 3:

            break

    return recommendations


# =====================================================
# EXPLAINABLE AI
# =====================================================

def explain_recommendation(
    skills,
    interest,
    career
):

    return f"""
### Why this career?

Your profile closely matches **{career}**.

### Skills Entered

{skills}

### Interest

{interest}

### The recommendation was generated using:

✅ TF-IDF Vectorization

✅ Cosine Similarity

The AI compares your skills and interests with career profiles
and recommends the closest matching careers based on profile similarity.
"""


# =====================================================
# SKILL GAP ANALYSIS
# =====================================================

def normalize_skill(
    skill
):

    skill = str(
        skill
    ).strip().lower()

    # Normalize common variations

    replacements = {

        "sql":
            "sql",

        "python":
            "python",

        "javascript":
            "javascript",

        "java script":
            "javascript",

        "node js":
            "node.js",

        "nodejs":
            "node.js",

        "node.js":
            "node.js",

        "html css":
            "html/css",

        "html/css":
            "html/css",

        "c plus plus":
            "c++",

        "c++":
            "c++",

        "data analytics":
            "data analysis"

    }

    return replacements.get(
        skill,
        skill
    )


def skill_gap_analysis(
    user_skills,
    career
):

    # Get required skills
    required = career_requirements.get(
        career,
        []
    )

    # Convert user input into individual skills
    user_skill_list = re.split(
        r"[,;|]",
        str(user_skills)
    )

    # Normalize user skills
    user_normalized = set(

        normalize_skill(
            skill
        )

        for skill in user_skill_list

        if str(skill).strip() != ""

    )

    # Normalize required skills
    required_normalized = [

        normalize_skill(
            skill
        )

        for skill in required

    ]

    # Find missing skills
    missing = [

        skill

        for skill, normalized in zip(
            required,
            required_normalized
        )

        if normalized not in user_normalized

    ]

    # Find matched skills
    matched = [

        skill

        for skill, normalized in zip(
            required,
            required_normalized
        )

        if normalized in user_normalized

    ]

    return (
        required,
        missing,
        matched
    )


# =====================================================
# SKILL MATCH PERCENTAGE
# =====================================================

def calculate_skill_match(
    required,
    matched
):

    if len(required) == 0:

        return 0.0

    percentage = (

        len(matched)
        /
        len(required)

    ) * 100

    return round(
        percentage,
        2
    )


# =====================================================
# LEARNING ROADMAP
# =====================================================

def generate_learning_plan(
    missing_skills
):

    roadmap = {}

    for skill in missing_skills:

        # Try exact match
        resource = learning_resources.get(
            skill
        )

        # Try case-insensitive match
        if resource is None:

            for key, value in learning_resources.items():

                if key.lower() == skill.lower():

                    resource = value

                    break

        # Default roadmap
        if resource is None:

            resource = (
                f"Learn {skill} "
                "from beginner to advanced level "
                "and build practical projects."
            )

        roadmap[skill] = resource

    return roadmap


# =====================================================
# CAREER DETECTION FOR CHATBOT
# =====================================================

def detect_career(
    question
):

    question = str(
        question
    ).lower().strip()

    # -------------------------------------------------
    # First check exact career names
    # -------------------------------------------------

    for career in career_list:

        if career.lower() in question:

            return career

    # -------------------------------------------------
    # Common career aliases
    # -------------------------------------------------

    aliases = {

        "ml engineer":
            "Machine Learning Engineer",

        "machine learning":
            "Machine Learning Engineer",

        "data science":
            "Data Scientist",

        "data scientist":
            "Data Scientist",

        "data analyst":
            "Data Analyst",

        "ai researcher":
            "AI Researcher",

        "ai research":
            "AI Researcher",

        "deep learning":
            "Deep Learning Engineer",

        "dl engineer":
            "Deep Learning Engineer",

        "software developer":
            "Software Engineer",

        "software development":
            "Software Engineer",

        "business analyst":
            "Business Analyst"

    }

    # -------------------------------------------------
    # Check aliases
    # -------------------------------------------------

    for alias, career_name in aliases.items():

        if alias in question:

            if career_name in career_list:

                return career_name

    return None


# =====================================================
# IMPROVED CAREER ADVISOR CHATBOT
# =====================================================

def chatbot(
    question
):

    question = str(
        question
    ).strip().lower()

    # -------------------------------------------------
    # Empty question
    # -------------------------------------------------

    if question == "":

        return (
            "Please enter a question."
        )

    # -------------------------------------------------
    # Detect career
    # -------------------------------------------------

    career = detect_career(
        question
    )

    # -------------------------------------------------
    # Career not detected
    # -------------------------------------------------

    if career is None:

        return (
            "I couldn't identify the career in your question.\n\n"

            "Please mention a career such as:\n\n"

            "• Machine Learning Engineer\n"
            "• Data Scientist\n"
            "• Data Analyst\n"
            "• AI Researcher\n"
            "• Deep Learning Engineer\n"
            "• Software Engineer\n"
            "• Business Analyst\n\n"

            "Example:\n"

            "What skills are required for Data Scientist?"
        )

    # -------------------------------------------------
    # Get career information
    # -------------------------------------------------

    required_skills = career_requirements.get(
        career,
        []
    )

    description = career_description.get(
        career,
        "Career information is not available."
    )

    # =================================================
    # SKILLS QUESTION
    # =================================================

    if (
        "skill" in question
        or "skills" in question
        or "require" in question
    ):

        if len(required_skills) > 0:

            skill_list = "\n".join(

                f"✅ {skill}"

                for skill in required_skills

            )

            return (

                f"### 🧠 Skills Required for {career}\n\n"

                f"{skill_list}\n\n"

                "💡 Tip: Learn the basic skills first "
                "and then build practical projects."

            )

        else:

            return (

                f"Skill information for {career} "
                "is currently not available."

            )

    # =================================================
    # ROADMAP QUESTION
    # =================================================

    elif (

        "roadmap" in question

        or "learn" in question

        or "learning" in question

        or "how to become" in question

        or "how can i become" in question

    ):

        if len(required_skills) > 0:

            roadmap = generate_learning_plan(
                required_skills
            )

            response = (

                f"### 📚 Learning Roadmap for {career}\n\n"

            )

            step = 1

            for skill, resource in roadmap.items():

                response += (

                    f"**Step {step}: {skill}**\n"

                    f"{resource}\n\n"

                )

                step += 1

            response += (

                "🎯 **Final Step**\n\n"

                "Build real-world projects, create a "
                "GitHub portfolio, complete internships "
                "and prepare for technical interviews."

            )

            return response

        else:

            return (

                f"Learning roadmap for {career} "
                "is currently not available."

            )

    # =================================================
    # SALARY QUESTION
    # =================================================

    elif (

        "salary" in question

        or "package" in question

        or "pay" in question

        or "earn" in question

    ):

        return (

            f"### 💰 Salary Information for {career}\n\n"

            "Salary depends on several factors:\n\n"

            "• Skills\n"
            "• Experience\n"
            "• Company\n"
            "• Location\n"
            "• Interview performance\n\n"

            "For accurate current salary information, "
            "check recent job postings and salary reports."

        )

    # =================================================
    # FUTURE SCOPE QUESTION
    # =================================================

    elif (

        "future" in question

        or "scope" in question

        or "demand" in question

        or "career growth" in question

    ):

        return (

            f"### 🚀 Future Scope of {career}\n\n"

            f"{career} offers opportunities across "
            "different industries.\n\n"

            "Career growth depends on:\n\n"

            "• Continuous learning\n"
            "• Technical skills\n"
            "• Project experience\n"
            "• Internships\n"
            "• Certifications\n"
            "• Professional experience"

        )

    # =================================================
    # PROJECT QUESTION
    # =================================================

    elif (

        "project" in question

        or "projects" in question

    ):

        return (

            f"### 💻 Project Ideas for {career}\n\n"

            "You can build:\n\n"

            f"• Beginner {career} project\n"

            f"• Intermediate {career} project\n"

            f"• Advanced {career} project\n\n"

            "💡 Choose projects that demonstrate "
            "the required skills for this career."

        )

    # =================================================
    # CAREER DESCRIPTION QUESTION
    # =================================================

    elif (

        "what is" in question

        or "about" in question

        or "description" in question

        or "role" in question

    ):

        return (

            f"### 👨‍💻 About {career}\n\n"

            f"{description}\n\n"

            "You can also ask me about:\n\n"

            "• Skills\n"
            "• Roadmap\n"
            "• Projects\n"
            "• Salary\n"
            "• Future Scope"

        )

    # =================================================
    # DEFAULT RESPONSE
    # =================================================

    else:

        return (

            f"I can help you learn about **{career}**.\n\n"

            "Try asking:\n\n"

            f"• What skills are required for {career}?\n"

            f"• Give me a roadmap for {career}.\n"

            f"• What projects should I build for {career}?\n"

            f"• What is the salary of {career}?\n"

            f"• What is the future scope of {career}?"

        )


# =====================================================
# GENERATE TEXT REPORT
# =====================================================

def generate_report(

    career,

    required,

    missing,

    percentage

):

    report = f"""

AI CAREER RECOMMENDATION REPORT
==========================================

Recommended Career:
{career}

------------------------------------------

Required Skills:
{", ".join(required)}

------------------------------------------

Missing Skills:
{", ".join(missing)}

------------------------------------------

Skill Match:
{percentage}%

------------------------------------------

Recommendation Method:
TF-IDF Vectorization
Cosine Similarity

------------------------------------------

Generated by:
AI Career Recommendation System

==========================================
"""

    return report
# =====================================================
# PART 3 — STREAMLIT USER INTERFACE
# =====================================================


# =====================================================
# APPLICATION HEADER
# =====================================================

st.title(
    "🎯 AI Career Recommendation System"
)

st.subheader(
    "Discover the Best Career using Artificial Intelligence"
)

st.write(
    "Get personalized career recommendations "
    "based on your skills and interests."
)


# =====================================================
# USER INPUT SECTION
# =====================================================

st.divider()

st.header(
    "📝 Enter Your Profile"
)


# -----------------------------------------------------
# Skills Input
# -----------------------------------------------------

skills = st.text_input(

    "Enter Your Skills",

    placeholder=(
        "Example: Python, SQL, Machine Learning"
    )

)


# -----------------------------------------------------
# Interest Input
# -----------------------------------------------------

interest = st.text_input(

    "Enter Your Interest",

    placeholder=(
        "Example: AI/ML, Data Science, Web Development"
    )

)


# =====================================================
# RECOMMENDATION BUTTON
# =====================================================

if st.button(
    "🚀 Get Career Recommendations"
):

    # -------------------------------------------------
    # Validate User Input
    # -------------------------------------------------

    if (

        skills.strip() == ""

        or

        interest.strip() == ""

    ):

        st.warning(

            "Please enter both your skills "
            "and your career interest."

        )

        st.stop()

    # =================================================
    # GET CAREER RECOMMENDATIONS
    # =================================================

    recommendations = recommend_from_input(

        skills,

        interest

    )

    # -------------------------------------------------
    # Check recommendations
    # -------------------------------------------------

    if len(recommendations) == 0:

        st.error(

            "Unable to generate career recommendations."

        )

        st.stop()

    # =================================================
    # TOP CAREER RECOMMENDATIONS
    # =================================================

    st.divider()

    st.header(
        "🏆 Top Career Recommendations"
    )

    # Display top 3 careers

    for i, (
        career,
        score
    ) in enumerate(

        recommendations,

        start=1

    ):

        st.subheader(

            f"{i}. {career}"

        )

        st.write(

            f"**Match Score : {score}%**"

        )

        description = career_description.get(

            career,

            "Career description not available."

        )

        st.info(

            description

        )

    # =================================================
    # BEST CAREER
    # =================================================

    top_career = recommendations[0][0]

    top_score = recommendations[0][1]

    # =================================================
    # CAREER MATCH
    # =================================================

    st.divider()

    st.header(
        "⭐ Career Match"
    )

    st.metric(

        "Confidence",

        f"{top_score}%"

    )

    # =================================================
    # EXPLAINABLE AI
    # =================================================

    st.divider()

    st.header(
        "🧠 Explainable AI"
    )

    explanation = explain_recommendation(

        skills,

        interest,

        top_career

    )

    st.markdown(

        explanation

    )

    # =================================================
    # SKILL GAP ANALYSIS
    # =================================================

    st.divider()

    st.header(
        "📊 Skill Gap Analysis"
    )

    # Get required, missing and matched skills

    required, missing, matched = (

        skill_gap_analysis(

            skills,

            top_career

        )

    )

    # =================================================
    # REQUIRED SKILLS
    # =================================================

    st.subheader(
        "✅ Required Skills"
    )

    if len(required) == 0:

        st.info(

            "Required skill information "
            "is not available for this career."

        )

    else:

        for skill in required:

            st.write(

                f"✔ {skill}"

            )

    # =================================================
    # MISSING SKILLS
    # =================================================

    st.subheader(
        "❌ Missing Skills"
    )

    if len(missing) == 0:

        st.success(

            "Excellent! You already have "
            "all the required skills."

        )

    else:

        for skill in missing:

            st.warning(

                skill

            )

    # =================================================
    # SKILL MATCH PERCENTAGE
    # =================================================

    percentage = calculate_skill_match(

        required,

        matched

    )

    st.subheader(
        "📈 Skill Match"
    )

    st.progress(

        percentage / 100

    )

    st.write(

        f"**{percentage}% Skills Matched**"

    )

    # =================================================
    # LEARNING ROADMAP
    # =================================================

    st.divider()

    st.header(
        "📚 Learning Roadmap"
    )

    # Generate roadmap only for missing skills

    roadmap = generate_learning_plan(

        missing

    )

    if len(roadmap) == 0:

        st.success(

            "🎉 No additional skills are required. "
            "Continue improving your existing skills "
            "through advanced projects."

        )

    else:

        for (

            skill,

            resource

        ) in roadmap.items():

            st.subheader(

                skill

            )

            st.info(

                resource

            )

    # =================================================
    # DOWNLOAD REPORT
    # =================================================

    st.divider()

    st.header(
        "📄 Career Recommendation Report"
    )

    report = generate_report(

        top_career,

        required,

        missing,

        percentage

    )

    st.download_button(

        label="📥 Download Career Report",

        data=report,

        file_name="AI_Career_Recommendation_Report.txt",

        mime="text/plain"

    )


# =====================================================
# CAREER ADVISOR CHATBOT
# =====================================================

st.divider()

st.header(
    "🤖 Career Advisor Chatbot"
)


st.write(

    "Ask questions about career skills, "
    "learning roadmap, projects, salary "
    "or future scope."

)


# =====================================================
# CHATBOT INPUT
# =====================================================

question = st.text_input(

    "Ask a question",

    placeholder=(
        "Example: What skills are required "
        "for Machine Learning Engineer?"
    )

)


# =====================================================
# CHATBOT BUTTON
# =====================================================

if st.button(
    "💬 Ask Chatbot"
):

    if question.strip() == "":

        st.warning(

            "Please enter a question."

        )

    else:

        response = chatbot(

            question

        )

        st.markdown(

            response

        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(

    "AI Career Recommendation System | "
    "Powered by TF-IDF and Cosine Similarity"

)

# =====================================================
# END OF APP.PY
# =====================================================
