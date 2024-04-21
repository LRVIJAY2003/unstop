import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def mock_interview_simulator(user_experience):

    X = np.random.rand(100, 10)  
    y = np.random.randint(2, size=100)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    rf_predictions = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, rf_predictions)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, rf_predictions))

    cm = confusion_matrix(y_test, rf_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    pass
