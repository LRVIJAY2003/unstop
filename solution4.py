from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def career_path_prediction(user_skills):

    features = [
        [5, 3, 1, 0],   # Sample feature vector 1
        [6, 3, 4, 1],   # Sample feature vector 2
        [6, 2, 4, 1],   # Sample feature vector 3
        [5, 4, 2, 0],   # Sample feature vector 4
        [7, 3, 6, 2]    # Sample feature vector 5
    ]

    labels = [0, 1, 1, 0, 2]  # Example labels for career paths (0, 1, 2 represent different paths)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    dt_classifier = DecisionTreeClassifier()

    dt_classifier.fit(X_train, y_train)

    dt_predictions = dt_classifier.predict(X_test)

    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print("Decision Tree Accuracy:", dt_accuracy)

    print("Classification Report for Decision Tree:")
    print(classification_report(y_test, dt_predictions))

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    rf_predictions = rf_classifier.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print("Random Forest Accuracy:", rf_accuracy)

    print("Classification Report for Random Forest:")
    print(classification_report(y_test, rf_predictions))

    plt.figure(figsize=(8, 6))
    cm_dt = sns.heatmap(confusion_matrix(y_test, dt_predictions), annot=True, cmap='Blues', fmt='g')
    cm_dt.set_title('Confusion Matrix - Decision Tree')
    cm_dt.set_xlabel('Predicted Labels')
    cm_dt.set_ylabel('True Labels')
    plt.show()

    plt.figure(figsize=(8, 6))
    cm_rf = sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, cmap='Blues', fmt='g')
    cm_rf.set_title('Confusion Matrix - Random Forest')
    cm_rf.set_xlabel('Predicted Labels')
    cm_rf.set_ylabel('True Labels')
    plt.show()

    pass
