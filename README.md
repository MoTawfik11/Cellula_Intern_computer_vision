# 🦷 Teeth Classification using TensorFlow  

## 📌 Project Overview  
This project focuses on teeth classification using Deep Learning and TensorFlow/Keras. The model is trained on a custom dataset of teeth images and aims to classify them into multiple categories (e.g., MC, OT, etc. depending on your dataset).

The goal of the project is to explore image preprocessing, model building, training, evaluation, and testing for medical/healthcare-related classification tasks. 

---

## ⚙️ Features  
- Image preprocessing and augmentation  
- Transfer learning using **EfficientNetB0** (or chosen CNN architecture)  
- Training with TensorFlow/Keras API  
- Accuracy and loss visualization  
- Evaluation on validation and test sets  
- Prediction on new images  


---

## 🛠️ Installation  

1. Clone this repository:  
```bash
git clone https://github.com/yourusername/teeth-classification.git
cd teeth-classification
```

---

## 💻 Usage  

### 🔹 Training the Model  

Open the notebooks/training.ipynb in Jupyter Notebook or JupyterLab and run all cells to train the model.

### 🔹 Testing on New Images  

Open the notebooks/evaluation.ipynb to test the trained model on validation/test datasets and to make predictions on new images.


## 📊 Results  
- Training Accuracy: 85%
- Test Accuracy: 84%  
- Example training curve:  
<img width="567" height="453" alt="output" src="https://github.com/user-attachments/assets/31bba70b-c220-402e-a042-12e8131ef4a8" />

<img width="567" height="453" alt="output2" src="https://github.com/user-attachments/assets/3ba34a8d-c2ed-48f4-826a-79beeb71b262" />

---

## 🧩 Challenges  
- Training for **50 epochs** took ~0.5 hours with limited GPU resources.  
- Model accuracy plateaued around **0.85**, which indicates need for:  
  - Hyperparameter tuning  
  - Data augmentation improvements  
  - Model architecture changes  

---

## 🚀 Future Work  
- Improve model performance using **pretrained CNNs** (EfficientNet, ResNet).  
- Apply **data augmentation** to increase dataset robustness.  
- Experiment with **learning rate schedulers** and **regularization techniques**.  
- Deploy the model as a **Flask/FastAPI web app** for real-time predictions.  

---

## 📜 Requirements  
- Python 3.8+  
- TensorFlow 2.x  
- Keras  
- NumPy  
- OpenCV  
- Matplotlib  

---

## ✨ Author  
👤 **Mohamed Tawfik**  
- Mechatronics Engineer | AI & Robotics Enthusiast  
- [LinkedIn]([https://www.linkedin.com](https://www.linkedin.com/in/mohamed-tawfik11/)) | [GitHub]([https://github.com](https://github.com/MoTawfik11))  
