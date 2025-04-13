import os
import pandas as pd

# Adjusted for your local directory
base_dir = r'D:\College\6th\deep learning\group\data\archive\AugmentedAlzheimerDataset'

MildDemented_dir = os.path.join(base_dir, 'MildDemented')
ModerateDemented_dir = os.path.join(base_dir, 'ModerateDemented')
NonDemented_dir = os.path.join(base_dir, 'NonDemented')
VeryMildDemented_dir = os.path.join(base_dir, 'VeryMildDemented')

filepaths = []
labels = []
dict_list = [MildDemented_dir, ModerateDemented_dir, NonDemented_dir, VeryMildDemented_dir]
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']

for i, folder in enumerate(dict_list):
    for f in os.listdir(folder):
        fpath = os.path.join(folder, f)
        filepaths.append(fpath)
        labels.append(class_labels[i])

Alzheimer_df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

print(Alzheimer_df.head())
print(Alzheimer_df["labels"].value_counts())
from PIL import Image
from collections import Counter

# Store all sizes
image_sizes = []

for path in Alzheimer_df['filepaths']:
    try:
        with Image.open(path) as img:
            image_sizes.append(img.size)  # (width, height)
    except Exception as e:
        print(f"Error opening {path}: {e}")

# Count and print unique sizes
size_counts = Counter(image_sizes)

print("Unique image sizes and their counts:")
for size, count in size_counts.items():
    print(f"Size: {size}, Count: {count}")
from sklearn.model_selection import train_test_split

train_images, test_images = train_test_split(Alzheimer_df, test_size=0.3, random_state=42)
train_set, val_set = train_test_split(train_images, test_size=0.2, random_state=42)

print("Train:", train_set.shape)
print("Validation:", val_set.shape)
print("Test:", test_images.shape)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Use simple rescaling (0–1 range), not MobileNetV2 preprocessing
image_gen = ImageDataGenerator(rescale=1./255)

# Train generator
train = image_gen.flow_from_dataframe(
    dataframe=train_set,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),  
    color_mode='rgb',
    class_mode="categorical",
    batch_size=32,
    shuffle=True
)

# Validation generator
val = image_gen.flow_from_dataframe(
    dataframe=val_set,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

# Test generator
test = image_gen.flow_from_dataframe(
    dataframe=test_images,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

# Get class labels (used in model output and evaluation)
classes = list(train.class_indices.keys())
print("Classes:", classes)
import matplotlib.pyplot as plt
import numpy as np

def show_sample_images(image_gen):
    classes = list(image_gen.class_indices.keys())
    images, labels = next(image_gen)
    plt.figure(figsize=(20, 20))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i+1)
        image = images[i]  
        plt.imshow(image)
        index = np.argmax(labels[i])
        plt.title(classes[index], color="green", fontsize=16)
        plt.axis('off')
    plt.show()

# Run it
show_sample_images(train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Number of output classes (based on your label categories)
num_classes = len(classes)

model = Sequential([
    # 1st Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # 2nd Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # 3rd Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten + Dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()
history = model.fit(
    train,
    validation_data=val,
    epochs=10  
)
loss, acc = model.evaluate(test)
print(f"Test Accuracy: {acc*100:.2f}%")
import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Get true labels and predictions
true_labels = test.classes
class_names = list(test.class_indices.keys())

# Predict using the model
pred_probs = model.predict(test)
pred_labels = np.argmax(pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Optional: Classification report
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))
