🩺 COVID-19 & Pneumonia Detection from X-Ray Images 🦠
This project implements a Deep Learning-based classification system for detecting COVID-19, Pneumonia, and Normal cases from chest X-ray images. It leverages CNNs with Transfer Learning (ResNet50, MobileNetV2) and Hyperparameter Optimization (HyperOpt) to improve model performance.

🚀 Key Features
✅ Deep Learning Model: Uses ResNet50 as a feature extractor with custom classification layers.
✅ Hyperparameter Tuning: Implements HyperOpt to optimize dropout rates, activation functions, and learning rates.
✅ Efficient Data Processing: Uses TensorFlow’s tf.data API for optimized data loading and augmentation.
✅ Model Deployment: A Streamlit web app for real-time X-ray image classification.
✅ MLflow Integration (Future Update): To track experiments and compare models.

📂 Dataset
The dataset consists of X-ray images classified into three categories:
📌 COVID-19
📌 Pneumonia
📌 Normal

The images are loaded directly using image_dataset_from_directory() from TensorFlow.

⚙ Project Workflow
1️⃣ Data Preprocessing & Augmentation
Images are resized to 224x224 and normalized.
Data Augmentation (Random Flip, Rotation, Zoom, Contrast) is applied.
TensorFlow's data pipeline (cache() & prefetch()) ensures faster loading.

2️⃣ Model Training & Hyperparameter Tuning
Uses ResNet50 with frozen base layers and custom Dense layers.
Hyperparameter tuning with HyperOpt (Tree-structured Parzen Estimator - TPE) optimizes:
Dropout rate
Number of neurons in Dense layers
Activation function
Optimizer (Adam, RMSprop)
Batch size & Epochs
Early stopping is used to prevent overfitting.
The best model is saved automatically (best_covid_model.h5).

3️⃣ Streamlit Web App Deployment
Users can upload an X-ray image via the app.
The trained model classifies the image and displays the result with confidence scores.
