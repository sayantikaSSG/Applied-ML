import pandas as pd

# Load the dataset
data_path = '/home/sayantika/Desktop/DS2sem4/Applied Machine Leaning/Applied-ML/train.csv'
df = pd.read_csv(data_path)

# Display the first few rows of the dataset to understand its structure
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

# 1. Text Vectorization
# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# 2. Training the SVM Model
# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')

# Create a pipeline that first vectorizes the text and then applies the SVM classifier
model_pipeline = make_pipeline(tfidf_vectorizer, svm_classifier)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['spam'], test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model_pipeline.score(X_test, y_test)

# 3. Saving the Model
# Define the path to save the model
model_path = '/home/sayantika/Desktop/DS2sem4/Applied Machine Leaning/Applied-ML/assignment 3/best_model.joblib'

# Save the model
joblib.dump(model_pipeline, model_path)

accuracy, model_path
