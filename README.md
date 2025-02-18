# ML_for_emotions

This project implements an SVM (Support Vector Machine) classifier to classify emotions into positive or negative based on EEG data. The dataset used is the [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset, a well-known database for analyzing human affective states using EEG and physiological signals. 

## What is an SVM?
A Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression. It works by finding the optimal hyperplane that best separates different classes in the feature space. SVMs are often used in research lab and clinical studies for emotional classification, because they are a good fit for small, high-dimensional classification problems. SVMs are more complex than basic models like logistic regression or k-nearest neighbors (KNN) but simpler than deep learning models, offering a good balance between interpretability and performance for structured data.

## DEAP (A Database for Emotion Analysis using Physiological Signals)
The DEAP dataset is a large-scale multimodal dataset containing EEG and physiological recordings from 32 participants watching 40 music videos, each lasting one minute. Participants rated each video on valence, arousal, dominance, and liking, which are used as emotional labels.

EEG Data: 32-channel EEG signals sampled at 128 Hz. The dataset includes code for preprocessing. 

## Pipeline of the Code

### Data Loading
The script loads EEG features from preprocessed .pkl (Pickle) files stored in Google Drive. Each file corresponds to a subject's processed EEG data.
```python
# Load data for each subject
subject_dirs = [f'/content/drive/MyDrive/Processed_data/sub{i:03d}.pkl' for i in range(6)]

# Initialize lists to store data and labels
all_eeg_data = []
all_video_labels = []

# Read EEG data from each subject
for subject_dir in subject_dirs:
    with open(subject_dir, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    eeg_data = preprocessed_data  # Assuming EEG features are already extracted
```
Explanation:
- EEG features are stored in multiple **`.pkl`** files, one for each subject.
- The script loads EEG feature vectors and their corresponding emotion labels.

### Data Preprocessing
To train an SVM model, the dataset needs to be prepared by:
- Extracting EEG features (if not already done).
- Normalizing the features to ensure consistency.
- Splitting data into training and testing sets.
```python
from sklearn.model_selection import train_test_split

# Convert lists to NumPy arrays
X = np.array(all_eeg_data)  # EEG feature vectors
y = np.array(all_video_labels)  # Emotion labels

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Explanation:
- EEG feature vectors (**`X`**) and emotion labels (**`y`**) are converted into NumPy arrays for compatibility with **`scikit-learn.
- Data is split into 80% training and 20% testing to evaluate the model's performance.

### Training the SVM Model
The Support Vector Machine is trained using the Radial Basis Function (RBF) kernel, which helps in non-linear classification.
```python
from sklearn.svm import SVC

# Initialize and train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
```
Explanation:
- **`SVC(kernel='rbf')`** specifies the Radial Basis Function kernel, ideal for non-linearly separable data.
- **`C=1.0`** controls the regularization (higher C = lower tolerance for misclassification).

### Evaluating the Model
The script evaluates model accuracy using **`accuracy_score`**.
```python
from sklearn.metrics import accuracy_score

# Predict on test set
y_pred = svm_model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```
Explanation:
- The trained model makes predictions on the test set.
- **`accuracy_score()`** calculates how well the model classifies emotions.

## Future Improvements
Feature Engineering: Extract more meaningful EEG features.
Different Classification Models: Try Deep Learning (CNN, LSTMs).
Real-time Emotion Detection: Implement a live EEG-based emotion recognition system.
