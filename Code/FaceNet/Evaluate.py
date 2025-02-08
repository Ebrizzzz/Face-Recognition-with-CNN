import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from keras_facenet import FaceNet
from matplotlib import pyplot as plt
import os

# Function to load faces and their labels
def load_dataset(dir):
    X, y = [], []
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            X.append(img_path)
            y.append(subdir)
    return np.array(X), np.array(y)

# Path to test dataset directory
test_dir = 'TrainData_Faces_Cropped_Test'

# Load the dataset
testX_paths, testy = load_dataset(test_dir)

print(f"Found {len(testX_paths)} testing images.")

# Initialize FaceNet model
embedder = FaceNet()

# Function to extract embeddings from images
def get_embeddings(paths, labels):
    embeddings = []
    valid_labels = []
    for i, (path, label) in enumerate(zip(paths, labels)):
        print(f"Processing image {i+1}/{len(paths)}")
        image = plt.imread(path)  # Load image using matplotlib
        detections = embedder.extract(image, threshold=0.95)
        if detections:
            embedding = detections[0]['embedding']
            embeddings.append(embedding)
            valid_labels.append(label)
        else:
            print(f"No face detected in image: {path}")
    return np.array(embeddings), np.array(valid_labels)

# Extract embeddings for the test set
emdTestX, testy_valid = get_embeddings(testX_paths, testy)

print(f"Extracted embeddings: Test set: {emdTestX.shape}")

# Load the saved models and encoders
model = joblib.load('face_classifier.pkl')
in_encoder = joblib.load('in_encoder.pkl')
out_encoder = joblib.load('out_encoder.pkl')

# Normalize the embeddings
emdTestX_norm = in_encoder.transform(emdTestX)

# Encode the labels
testy_enc = out_encoder.transform(testy_valid)

# Make predictions
yhat_test = model.predict(emdTestX_norm)

# Generate evaluation metrics
print("Classification Report:")
print(classification_report(testy_enc, yhat_test, target_names=out_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(testy_enc, yhat_test))

# Optional: Plot confusion matrix (requires seaborn)
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(testy_enc, yhat_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=out_encoder.classes_, yticklabels=out_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
