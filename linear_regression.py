import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
aura = pd.read_csv('/home/aura/Documents/2.2/datascience/accidents.csv')

print("Dataset loaded successfully!")
print(f"Shape: {aura.shape}")
print(f"Columns: {aura.columns.tolist()}")


print(f"\nAccident severity value counts:")
print(aura['Accident_severity'].value_counts())

# Define variables
# Dependent variable (what we predict): Accident_severity
# Independent variables (factors that influence severity)
X = pd.get_dummies(aura[['Age_band_of_driver', 'Driving_experience', 'Road_surface_type', 'Light_conditions']])
y = aura['Accident_severity']

print(f"\nFeatures after encoding: {X.shape[1]} columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Results:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Saving
with open('accident_severity_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'accident_severity_model.pkl'")

# Example prediction
print("==== EXAMPLE PREDICTION ====")


# Create a hypothetical accident scenario
sample_data = {
    'Age_band_of_driver': '18-30',
    'Driving_experience': '2-5yr', 
    'Road_surface_type': 'Asphalt roads',
    'Light_conditions': 'Darkness'
}

sample_df = pd.DataFrame([sample_data])
sample_encoded = pd.get_dummies(sample_df)

# Align columns with training data (fill missing columns with 0)
sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)

# predicting :
prediction = model.predict(sample_encoded)[0]
print("Hypothetical Accident Scenario:")
print(f"- Age: 18-30")
print(f"- Experience: 2-5 years") 
print(f"- Road: Asphalt roads")
print(f"- Light: Darkness")
print(f"Predicted accident severity: {prediction:.2f}")

