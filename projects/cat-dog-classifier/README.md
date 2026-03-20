# 🐶🐱 Cats vs Dogs Image Classification using CNN

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images of **cats and dogs**.
It is a beginner-friendly deep learning project focused on binary image classification.

---

## 📂 Dataset

* **Source:** Kaggle
* **Dataset Name:** Dog and Cat Classification Dataset
* **Author:** bhavikjikadara
* **Link:** https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

### 📥 Dataset Usage

The dataset was **manually downloaded from Kaggle** and extracted locally.

### 📁 Dataset Structure

```id="hlszy6"
dataset/
├── cats/
├── dogs/
```

* **cats/** → contains all cat images 🐱
* **dogs/** → contains all dog images 🐶

---

## 🛠️ Technologies Used

* Python 🐍
* PyTorch 🔥
* Torchvision

---

## ⚙️ Features

* Image preprocessing using `torchvision.transforms`
* CNN-based architecture
* Training loop with loss calculation
* Accuracy evaluation
* Simple and clean implementation

---

## 🧠 Model Architecture

* Convolution Layer → ReLU → MaxPooling
* Fully Connected Layer
* Output Layer (2 classes: Cat & Dog)

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision
```

### 2️⃣ Download Dataset

* Go to the Kaggle dataset link
* Download and extract it
* Keep folder structure as shown above

### 3️⃣ Update Dataset Path

```python
path = "your/dataset/path"
```

Example:

```python
path = "C:/Users/ayush/Desktop/coding/ML/archive (2)/PetImages/dataset"
```

---

### 4️⃣ Run the Code

```bash
python train.py
```

---

## 📊 Output

* Training loss printed for each epoch
* Final accuracy displayed

### Example Output

```id="t0zwnj"
Epoch 1, Loss: 0.6932, Accuracy: 52.34%
Epoch 2, Loss: 0.5127, Accuracy: 74.18%
Epoch 3, Loss: 0.4215, Accuracy: 81.67%
...
Final Accuracy: 85.23%
```

---

## ⚠️ Notes

* Model is trained on CPU
* Same dataset is used for training and testing (for simplicity)
* Accuracy may be higher than real-world performance

---

## 🔮 Future Improvements

* Add train-validation split
* Use GPU for faster training
* Apply data augmentation
* Use transfer learning (ResNet, VGG)
* Build a web app for predictions

---

## 👨‍💻 Author

**Ayush Agarwal**

---

## ⭐ Acknowledgements

* Kaggle for providing the dataset
* PyTorch for the deep learning framework

---

## 📌 Conclusion

This project demonstrates a **basic CNN implementation for image classification** and serves as a strong starting point for more advanced deep learning projects.

---
