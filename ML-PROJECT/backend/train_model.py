import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

# Load the dataset
df = pd.read_csv('d:\\study college\\3rd year\\ML-PROJECT\\studataset.csv')

# Create a synthetic 'depression' column
df['depression'] = ((df['failures'] > 0) | (df['absences'] > 15) | (df['G3'] < 8)).astype(int)

# Select features and target
features = ['failures', 'absences', 'goout', 'Dalc', 'Walc', 'studytime', 'health', 'G3']
target = 'depression'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model next to this script so the API can find it
out_path = Path(__file__).parent / 'model.pkl'
joblib.dump(model, out_path.as_posix())

print("Model trained and saved as model.pkl")
