import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib


aura = pd.read_csv("/home/aura/Documents/2.2/datascience/diabetes.csv")


x = aura.drop("Outcome", axis=1)#questions-how old is the person etc
y = aura["Outcome"]#answers-does the person have diabetes?

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# desicion maker
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(x_train, y_train)

# --- Save the trained model 
joblib.dump(clf, 'decision_tree_model.pkl')

# --- Make predictions on test data ---
y_pred = clf.predict(x_test)

# --- Evaluate the model ---
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Visualize the Decision Tree ---
plt.figure(figsize=(12,8))
plot_tree(
    clf,
    filled=True,
    feature_names=aura.columns[:-1],
    class_names=["No Diabetes", "Diabetes"]
)
plt.show()

# --- Load model and predict on new data ---
loaded_model = joblib.load('decision_tree_model.pkl')

new_data = pd.DataFrame({
    'Pregnancies': [2, 4, 1, 5],
    'Glucose': [120, 150, 90, 200],
    'BloodPressure': [70, 80, 60, 90],
    'SkinThickness': [20, 25, 15, 30],
    'Insulin': [79, 130, 50, 180],
    'BMI': [25.0, 30.5, 22.0, 35.5],
    'DiabetesPedigreeFunction': [0.5, 0.7, 0.0, 0.6],
    'Age': [25, 45, 22, 50]
})

predictions = loaded_model.predict(new_data)
diabetes_map = {0: 'No Diabetes', 1: 'Diabetes'}

for i, prediction in enumerate(predictions):
    label = diabetes_map[prediction]
    print(f"Data {i+1}: Predicted Outcome → {label}")
