import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("pokemon.csv")

# Fill missing values with 'None'
data.fillna("None", inplace=True)

# Convert all string columns to lowercase
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.lower()

# Encode abilities using One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = encoder.fit_transform(data[['Ability1', 'Ability2', 'HiddenAbility']])

# Encode Pokémon names as labels
y = data['Pokemon'].astype('category').cat.codes
pokemon_classes = dict(enumerate(data['Pokemon'].astype('category').cat.categories))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test, y_pred,
    labels=np.unique(y_test),
    target_names=[pokemon_classes[i] for i in np.unique(y_test)]
)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Save the trained model and encoders
joblib.dump(model, "pokemon_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(pokemon_classes, "pokemon_classes.pkl")

print("Model training complete. Saved as 'pokemon_model.pkl'")
