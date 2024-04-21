from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def enhanced_job_matching(job_requirements, candidate_profile):

    features = [[0, 0], [1, 1], [2, 2], [3, 3]]  
    labels = [0, 1, 2, 3]  

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    predictions = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    pass
