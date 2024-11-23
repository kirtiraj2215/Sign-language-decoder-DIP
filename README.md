# **American Sign Language (ASL) Gesture Recognition Using CNN**

Welcome to the repository for our ASL Gesture Recognition project! This project uses a Convolutional Neural Network (CNN) to identify American Sign Language (ASL) gestures from images. With 36 distinct classes representing letters and numbers, this model is designed to help bridge communication gaps using sign language recognition.

---

## **About the Project**

This project leverages deep learning techniques to create a robust system for identifying ASL gestures. The goal is to provide an accessible and scalable solution for real-time gesture recognition. Using TensorFlow and Keras, we’ve designed a CNN that processes input images and predicts the corresponding ASL class with high accuracy. 

### **What You’ll Find Here**
- An end-to-end pipeline for training, evaluating, and visualizing results.
- A well-structured, easy-to-use codebase designed for further exploration and experimentation.
- Detailed results and metrics showcasing the performance of the model.

---

## **Getting Started**

Here’s how you can set up this project on your local machine:

### **1. Clone the Repository**
```bash
git clone https://github.com/kirtiraj2215/Sign-language-decoder-DIP.git
cd Sign-language-decoder-DIP
```

### **2. Set Up the Environment**
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### **3. Prepare the Dataset**
- Place your labeled dataset (with 36 ASL gesture classes) in the `datasets/` folder.
- Ensure the data is split into training and validation sets.

---

## **Dependencies & Requirements**

### **Libraries and Tools**
- **TensorFlow/Keras**: To design and train the CNN model.
- **Matplotlib/Seaborn**: For plotting graphs and visualizing results.
- **Sklearn**: For calculating performance metrics like accuracy, F1-score, and confusion matrix.
- **NumPy/Pandas**: For efficient data handling and preprocessing.

### **System Requirements**
- Python 3.7 or higher
- GPU-enabled system (NVIDIA GPU recommended) for faster training
- Minimum: 20 GB of storage and 8 GB of RAM

### **Dataset**
A labeled image dataset with 36 ASL gesture classes, including both letters (A-Z) and numbers (0-9).

---

## **How to Use**

### **Train the Model**
Run the following command to start training:
```bash
python main.py --train
```

### **Evaluate the Model**
To test the model’s performance:
```bash
python main.py --evaluate
```

### **Visualize Results**
Training logs, accuracy plots, and confusion matrices will be saved in the `outputs/` folder for easy analysis.

---

## **Key Features**
- **CNN Architecture**: A custom-built CNN tailored for gesture recognition with layers for convolution, pooling, and dropout to enhance performance.
- **Robust Evaluation**: Metrics like accuracy, precision, and recall provide a clear view of the model's effectiveness.
- **Scalability**: Designed to accommodate more classes or larger datasets with minimal changes.

---

## **Results**
- Achieved high accuracy on test data with a well-balanced confusion matrix.
- Effective visualization of training progress and evaluation metrics.
- A reliable architecture that generalizes well to unseen data.

---

## **Project Structure**
```
├── datasets/               # Dataset storage
├── models/                 # Saved trained model
├── outputs/                # Visualizations and logs
├── requirements.txt        # List of dependencies
├── README.md               # Documentation of the github
└── main.py                 # Main script for training and evaluation
```

---

## **Contributing**

We welcome contributions to improve this project. Whether it’s fixing bugs, adding features, or improving documentation, feel free to get involved! To contribute:
1. Fork this repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a pull request.

---

## **Acknowledgments**
This project wouldn’t have been possible without the ASL dataset contributors and the open-source community. Special thanks to everyone who provided resources and support for this initiative.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
