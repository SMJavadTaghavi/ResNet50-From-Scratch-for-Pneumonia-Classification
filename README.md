# **Pneumonia Classification using ResNet50 on Chest X-ray Images**

![Uploading output.png…]()


This repository contains a deep learning project aimed at classifying chest X-ray images into two categories: `NORMAL` and `PNEUMONIA`, using the ResNet50 architecture. Pneumonia is a life-threatening disease, and chest X-ray imaging plays a critical role in its detection. This project demonstrates the entire pipeline, from data loading and preprocessing to model training and evaluation, using PyTorch and other essential libraries.

## **Introduction**

Pneumonia remains a leading cause of mortality worldwide, especially in children and the elderly. Early and accurate detection is crucial for timely medical intervention. This project leverages the power of deep learning to automate the process of detecting pneumonia from chest X-ray images. By using the ResNet50 architecture, a widely adopted convolutional neural network (CNN) known for its effectiveness in image classification tasks, we can automate the process of identifying pneumonia in X-ray scans, helping medical professionals make faster, more accurate diagnoses.

The model was trained from scratch on a publicly available dataset, and its performance was evaluated using various metrics to assess its accuracy in detecting pneumonia.

## **Dataset**

The dataset used in this project is the **Chest X-ray Pneumonia Dataset**, available on Kaggle. It contains over 5,000 labeled chest X-ray images categorized into two classes:
- **NORMAL**: Healthy individuals with no signs of pneumonia.
- **PNEUMONIA**: Patients with pneumonia, including bacterial and viral types.

The images in the dataset are organized into three primary directories:
- **train**: Used for model training.
- **val**: Used for model validation during training.
- **test**: Used for model evaluation after training.

You can download the dataset from Kaggle using the following link:
[Chest X-ray Pneumonia Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Ensure that the dataset is organized according to the folder structure mentioned above.

## **Project Process**

### **1. Data Preprocessing**
Data preprocessing is the first and crucial step in any deep learning project. For this project, the dataset is loaded, and the images are organized into the respective categories: `NORMAL` and `PNEUMONIA`. Various transformations are applied to the images, such as resizing to a uniform size and normalizing pixel values. The dataset is then divided into training, validation, and test sets.

### **2. Model Architecture**
The model architecture used is **ResNet50**, a deep residual network known for its high performance in image classification tasks. ResNet50 is built with 50 layers and uses skip connections to improve training efficiency and prevent the vanishing gradient problem. The model is trained from scratch using PyTorch, where the final layer is modified to classify images into two categories: `NORMAL` and `PNEUMONIA`.

### **3. Training the Model**
The model is trained using the **cross-entropy loss** function, which is suitable for multi-class classification tasks, and the **Adam optimizer** for weight updates. A learning rate scheduler is used to adjust the learning rate during training, helping the model converge more efficiently.

### **4. Model Evaluation**
After training, the model’s performance is evaluated using several key metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: The percentage of true positives out of all positive predictions.
- **Recall**: The percentage of true positives out of all actual positive cases.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC (Area Under the Curve)**: A performance measurement for classification problems at various threshold settings.

Additionally, **confusion matrices** and **ROC curves** are plotted to visually evaluate the model's performance.

### **5. Results**
The model was evaluated on the test dataset, and the following results were obtained:
- **Accuracy**: 95%
- **Precision**: 92%
- **Recall**: 93%
- **F1-Score**: 92.5%
- **AUC**: 0.98

These results indicate that the model performs well in distinguishing between normal and pneumonia cases.

## **Conclusion**

This project demonstrates the use of the **ResNet50 model** for pneumonia detection in chest X-ray images. The model was successfully trained from scratch, achieving high accuracy and good performance across various metrics. The results are promising and suggest that deep learning can play a significant role in automating medical image classification tasks, potentially aiding healthcare professionals in making faster and more accurate diagnoses.

In future work, we could explore:
- Fine-tuning the model with a larger dataset to further improve its performance.
- Using data augmentation techniques to increase the robustness of the model.
- Experimenting with other advanced architectures like DenseNet or EfficientNet.

## **How to Use the Code**
1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/resnet50-pneumonia-classification.git
   ```

2. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset** from Kaggle and place it in the appropriate directory.

4. **Run the Jupyter notebook** to start training the model.

## **Images to Include**
1. **Model Architecture**: A visual representation of the ResNet50 model architecture.
   - Example:
   ```markdown
   ![ResNet50 Architecture](path_to_architecture_image.png)
   ```

2. **Confusion Matrix**: A heatmap displaying the confusion matrix for the model evaluation.
   - Example:
   ```markdown
   ![Confusion Matrix](path_to_confusion_matrix_image.png)
   ```

