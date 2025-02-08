import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.regularizers import l2  

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

train_dir = "Cropprd/"
test_dir = "Cropped__Test/"

# Using ImageDataGenerator for data augmentation
generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,             # Shear transformation
    zoom_range=0.2,               # Random zoom
    rotation_range=20,            # Random rotation in the range [-20, 20] degrees
    width_shift_range=0.1,        # Random horizontal shift
    height_shift_range=0.1,       # Random vertical shift
    horizontal_flip=True,        # Random horizontal flip
    vertical_flip=True,          # Random vertical flip
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    validation_split=0.1          # 10% of the data will be used for validation
)


# Load and split the data into training and validation sets
train_ds = generator.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    subset="training"  # This is for training data
)

val_ds = generator.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    subset="validation"  # This is for validation data
)

# Load the test data from the test directory
test_generator = ImageDataGenerator(rescale=1./255)
test_ds = test_generator.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False  
)

# Get the list of classes
classes = list(train_ds.class_indices.keys())
print(classes)

# Load VGG16 model with pre-trained ImageNet weights
vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze the layers
for layer in vgg.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(vgg.output)
x = Dense(4096, activation='relu')(x)   
x = Dropout(0.5)(x)
x = Dense(1023, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(3, activation='softmax')(x) 
model = Model(inputs = vgg.input, outputs = x)

# Create the model
model = Model(inputs=vgg.input, outputs=x)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(train_ds, epochs=30, validation_data=val_ds)

# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model on the training set
train_loss, train_accuracy = model.evaluate(train_ds)
print(f"Training Accuracy: {train_accuracy*100: .2f}")

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_ds)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')


# Predict on the test set
test_ds.reset()
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_ds.classes
class_labels = list(test_ds.class_indices.keys())

# Compute and print classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Compute and plot confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save('face_recognition_model_vgg16_cropped.keras')
