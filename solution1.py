import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def personalized_job_recommendations(user_id):

    user_data = {
        'user_id': [1, 2, 3],
        'search_history': ['data science', 'machine learning', 'python programming']
    }

    job_data = {
        'job_id': [101, 102, 103],
        'job_title': ['data scientist', 'machine learning engineer', 'python developer'],
        'job_description': ['...data science role description...', '...ML engineer role description...', '...python developer role description...']
    }

    hackathon_data = {
        'hackathon_id': [201, 202, 203],
        'hackathon_title': ['AI Hackathon', 'Data Science Challenge', 'Python Coding Contest'],
        'hackathon_description': ['...AI Hackathon description...', '...Data Science Challenge description...', '...Python Coding Contest description...']
    }

    user_df = pd.DataFrame(user_data)
    job_df = pd.DataFrame(job_data)
    hackathon_df = pd.DataFrame(hackathon_data)

    all_content = list(job_df['job_description']) + list(hackathon_df['hackathon_description'])

    tfidf_vectorizer = TfidfVectorizer()
    content_tfidf = tfidf_vectorizer.fit_transform(all_content)
    user_search_tfidf = tfidf_vectorizer.transform(user_df['search_history'])

    cos_sim_matrix = cosine_similarity(user_search_tfidf, content_tfidf)

    recommended_content = []
    for i, user_id in enumerate(user_df['user_id']):
        recommended_job_index = cos_sim_matrix[i][:len(job_df)].argmax()
        recommended_hackathon_index = cos_sim_matrix[i][len(job_df):].argmax()
        recommended_job = job_df.iloc[recommended_job_index]
        recommended_hackathon = hackathon_df.iloc[recommended_hackathon_index]
        recommended_content.append({'user_id': user_id, 'recommended_job': recommended_job['job_title'], 'recommended_hackathon': recommended_hackathon['hackathon_title']})

    for rec_content in recommended_content:
        print(f"User {rec_content['user_id']} - Recommended Job: {rec_content['recommended_job']}, Recommended Hackathon: {rec_content['recommended_hackathon']}")

    pass
