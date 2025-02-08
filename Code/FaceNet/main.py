import numpy as np
import os
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from random import choice
from matplotlib import pyplot as plt
import joblib

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

# Paths to dataset directories
train_dir = 'TrainData_Faces_Cropped'
test_dir = 'TrainData_Faces_Cropped_Test'

# Load the dataset
trainX_paths, trainy = load_dataset(train_dir)
testX_paths, testy = load_dataset(test_dir)

print(f"Found {len(trainX_paths)} training images and {len(testX_paths)} testing images.")

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

# Extract embeddings for train and test sets
emdTrainX, trainy_valid = get_embeddings(trainX_paths, trainy)
emdTestX, testy_valid = get_embeddings(testX_paths, testy)

print(f"Extracted embeddings: Train set: {emdTrainX.shape}, Test set: {emdTestX.shape}")

# Save embeddings
np.savez_compressed('celebrity-faces-embeddings.npz', emdTrainX, trainy_valid, emdTestX, testy_valid)

# Train SVM classifier
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy_valid)
trainy_enc = out_encoder.transform(trainy_valid)
testy_enc = out_encoder.transform(testy_valid)

model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainy_enc)

# Evaluate the model
yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)
score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)

print(f'Accuracy: train={score_train*100:.3f}, test={score_test*100:.3f}')

# Save the trained SVM model and the encoders
joblib.dump(model, 'face_classifier.pkl')
joblib.dump(in_encoder, 'in_encoder.pkl')
joblib.dump(out_encoder, 'out_encoder.pkl')

# Randomly select a face from the test set for prediction
selection = choice([i for i in range(len(testX_paths))])
random_face_path = testX_paths[selection]
random_face = plt.imread(random_face_path)
random_face_emd = emdTestX_norm[selection]
random_face_class = testy_enc[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

sample = np.expand_dims(random_face_emd, axis=0)
yhat_class = model.predict(sample)
yhat_prob = model.predict_proba(sample)

class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

print(f'Predicted: {predict_names[0]} ({class_probability:.3f}%)')
print(f'Expected: {random_face_name[0]}')

plt.imshow(random_face)
plt.title(f'{predict_names[0]} ({class_probability:.3f}%)')
plt.show()