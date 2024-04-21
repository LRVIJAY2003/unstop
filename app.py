from flask import Flask, render_template, request
from solutions import solution1, solution2, solution3, solution4, solution5, solution6

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/job_recommendations/<int:user_id>')
def job_recommendations(user_id):
    recommendations = solution1.personalized_job_recommendations(user_id)
    return render_template('job_recommendations.html', recommendations=recommendations)

@app.route('/job_matching', methods=['POST'])
def job_matching():
    job_requirements = request.form['job_requirements']
    candidate_profile = request.form['candidate_profile']
    matched_job = solution2.enhanced_job_matching(job_requirements, candidate_profile)
    return render_template('job_matching.html', matched_job=matched_job)

@app.route('/market_insights')
def market_insights():
    insights = solution3.real_time_market_insights()
    return render_template('market_insights.html', insights=insights)

@app.route('/career_prediction', methods=['POST'])
def career_prediction():
    user_skills = request.form['user_skills']
    predicted_career_path = solution4.career_path_prediction(user_skills)
    return render_template('career_prediction.html', predicted_career_path=predicted_career_path)

@app.route('/mock_interview', methods=['POST'])
def mock_interview():
    user_experience = request.form['user_experience']
    mock_interview_feedback = solution5.mock_interview_simulator(user_experience)
    return render_template('mock_interview.html', mock_interview_feedback=mock_interview_feedback)

@app.route('/interview_analysis', methods=['POST'])
def interview_analysis():
    interview_responses = request.form['interview_responses']
    interview_analysis_insights = solution6.real_time_interview_analysis(interview_responses)
    return render_template('interview_analysis.html', interview_analysis_insights=interview_analysis_insights)

if __name__ == '__main__':
    app.run(debug=True)
