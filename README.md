# Alzheimer-s-detection
Alzheimer's Disease Image Classification
This project uses deep learning to classify MRI images into four categories of Alzheimer's disease progression:

	Non Demented
	Very Mild Demented
	Mild Demented
	Moderate Demented

🧠 Dataset
The dataset is structured with one folder per class inside:

	AugmentedAlzheimerDataset/
	├── MildDemented/
	├── ModerateDemented/
	├── NonDemented/
	└── VeryMildDemented/
Each folder contains preprocessed MRI brain scan images related to the corresponding class.

📊 Class Distribution

	Class	Count
	Non Demented	9600
	Mild Demented	8960
	Very Mild Demented	8960
	Moderate Demented	6464
 
⚙️ Preprocessing

	Image paths and labels are collected and stored in a pandas DataFrame.
	Image sizes are analyzed to ensure consistency.
	Data is split into:
		70% Train
		21% Validation (20% of train)
		30% Test
	Images are resized to 224x224 and rescaled to [0, 1].

🧪 Image Generators

	Keras ImageDataGenerator is used for:
		Rescaling
		Batching
		Shuffling (for training only)

Generators are created for train, validation, and test datasets.

🧱 Model Architecture
A basic Convolutional Neural Network (CNN):

	Sequential([
	    Conv2D(32), MaxPooling2D(),
	    Conv2D(64), MaxPooling2D(),
	    Conv2D(128), MaxPooling2D(),
	    Flatten(),
	    Dense(128), Dropout(0.5),
	    Dense(4, activation='softmax')
	])
Loss: Categorical Crossentropy
Optimizer: Adam
Metric: Accuracy

📈 Training

	Model is trained for 10 epochs.
	Both training and validation accuracy are tracked.


📁 Project Structure

	├── data/
	│   └── archive/
	│       └── AugmentedAlzheimerDataset/
	├── main_script.py
	├── README.md
 
🧪 Requirements

	Python 3.8+
	TensorFlow
	pandas
	numpy
	scikit-learn
	matplotlib
	seaborn
	Pillow

Install all packages via:

	pip install -r requirements.txt


✅ Output

	Training and validation accuracy plots
	Test accuracy percentage
	Confusion matrix and classification report
