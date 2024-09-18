import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE
from flask import Flask, request, jsonify
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
import os

# Function to scrape text from a given URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        print(f"Error while scraping: {str(e)}")
        return None

# Load the dataset
df = pd.read_csv("dataset.csv")

# Split the dataset into features (X) and labels (y)
X = df['text'].values
y = df['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization for the training and testing sets
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Oversample the minority class using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Feature engineering pipeline for the new samples
feature_engineering_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=5000)),
    ('scaler', StandardScaler(with_mean=False))
])

# Transform the training and testing sets using the feature engineering pipeline
X_train_features = feature_engineering_pipeline.fit_transform(X_train).toarray()
X_test_features = feature_engineering_pipeline.transform(X_test).toarray()

# Build and compile the neural network model
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(X_train_features.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train_features, y_train, epochs=20, batch_size=32, validation_data=(X_test_features, y_test))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test_features, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the trained model
model.save('dark_pattern_detector_model.h5')

# URL to check for new samples
url_to_check = "https://www.viit.ac.in/"

# Scrape text from the URL
new_samples = [scrape_text_from_url(url_to_check)]

# Transform the new samples using the feature engineering pipeline
new_samples_features = feature_engineering_pipeline.transform(new_samples).toarray()

# Make predictions on the new samples
predictions = model.predict(new_samples_features)

# Set a threshold for classification
threshold = 0.30
predicted_classes = [1 if prediction > threshold else 0 for prediction in predictions]

# Display results
for i, sample in enumerate(new_samples):
    print(f"Sample: {sample}")
    print(f"Predicted Class: {'Deceptive' if predicted_classes[i] == 1 else 'Non-Deceptive'}")
    print("----------------------")

# Flask App
app = Flask(__name__)

REPO_ID = "pirocheto/phishing-url-detection"
FILENAME = "model.onnx"
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

@app.route("/")
def hello():
    return "Hello"

@app.route("/check-url", methods=["POST"])
def check_url():
    try:
        url = request.json.get("url")
        print("Received URL:", url)
        if not url:
            return "No URL provided in the request.", 400

        result = sess.run(None, {"inputs": np.array([url], dtype="str")})[1]

        return jsonify({"prediction": f"{result[0][1] * 100:.2f} %"})

    except Exception as e:
        print("Error:", e)
        return "Internal server error.", 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
