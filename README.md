Butterfly model - https://drive.google.com/drive/folders/1jyiya0IboU6vkIQDixn4VrGIf1NkuzSL?usp=drive_link

BUTTERFLY SPECIES CLASSIFICATION
Using Transfer Learning (MobileNetV2)
Deep Learning Computer Vision | Python
Yousha Tamzid
Department of Computer Science & Engineering
Abstract
This project presents an end-to-end deep learning pipeline for classifying butterfly species from images. Using
transfer learning with MobileNetV2 pre-trained on ImageNet, the model was trained and fine-tuned on 6,499
labeled images spanning 75 butterfly species. The final model achieved a training accuracy of 95.56% and a
validation accuracy of 88.54% after a two-phase strategy of feature extraction followed by selective fine-tuning.
The model was deployed as a Streamlit web application for real-time species prediction.
1. Introduction
Butterfly species identification is a challenging task due to high visual similarity between species and the large
number of distinct classes. Traditionally this relied on expert entomologists, making it time-consuming. This
project leverages Transfer Learning, adapting a model pre-trained on a large dataset to a new domain-specific task
to build a robust butterfly classifier with a Streamlit web interface.
2. Dataset
Attribute Details
Total Images 6,499
Number of Classes 75 butterfly species
Image Format JPEG/JPG
Input Resolution 224 x 224 pixels
Source Format ZIP archive with CSV labels
1
Sample species include MONARCH, ADONIS, BROWN SIPROETA, SOUTHERN DOGFACE, and GREEN
CELLED CATTLEHEART. Labels were encoded numerically using sklearn's LabelEncoder for neural
network compatibility.
3. Methodology
3.1 Data Preprocessing
Images extracted from ZIP archive and read using OpenCV (cv2).
Converted BGR → RGB and resized to 224 × 224 px for MobileNetV2 compatibility.
Pixel values scaled to [-1, 1] via mobilenet_v2.preprocess_input.
Dataset split: 80% training / 20% validation using stratified sampling.
3.2 Data Augmentation
RandomFlip (horizontal)
Random Rotation (±10%)
RandomZoom (±10%)
3.3 Model Architecture
MobileNetV2 (frozen base, ImageNet weights)
→ GlobalAverage Pooling2D
→ Dropout (0.4)
→ Dense (75, activation='softmax')
3.4 Training Strategy
Phase Epochs LR Frozen Layers
Feature Extraction 30 0.0001 All MobileNetV2 layers
Fine-Tuning 10 0.00001 All except last 30 layers
EarlyStopping monitored val_loss with patience = 5 to prevent overfitting.
•
•
•
•
•
•
•
•
•
•
•
2
4. Results
Phase Epochs Train Acc Val Acc
Feature Extraction 30 91.98% 87.00%
Fine-Tuning 10 95.56% 88.54%
The model showed consistent improvement across both phases, with validation accuracy stabilising above 88%
after fine-tuning. Dropout regularisation and data augmentation together prevented significant overfitting
throughout training.
5. Web Application
Upload any butterfly image (JPG/PNG).
Real-time species prediction with confidence score.
Visual confidence progress bar.
Uncertainty warning when confidence falls below 30%.
Component Technology
Frontend/UI Streamlit
Model Inference TensorFlow / Keras
Image Processing Pillow, NumPy
Model File butterfly_model.h5
Label Mapping class_names.json
pip install streamlit tensorflow pillow numpy
streamlit run app.py
6. Project Structure
butterfly-classifier/
├── butterflys.ipynb
├── app.py
├── butterfly_model.h5
•
•
•
•
3
├── class_names.json
└── archive.zip
7. Technologies Used
Python 3.12: Core programming language
TensorFlow / Keras: Model building and training
MobileNetV2: Pre-trained base model (ImageNet weights)
OpenCV (cv2): Image loading and preprocessing
scikit-learn: Label encoding, train/test split
Pandas/NumPy: Data manipulation
Matplotlib / Seaborn: Visualization and plotting
Streamlit: Web application deployment
8. Conclusion
This project demonstrates that transfer learning with MobileNetV2 is an effective approach for butterfly species
classification, achieving over 88% validation accuracy across 75 distinct species with approximately 6,500 images.
The two-phase training strategy first training only the classification head, then selectively fine-tuning the deeper
layers proved reliable and computationally efficient. The Streamlit web application makes the model accessible to
non-technical users through a simple image-upload interface.
9. Future Work
Expand the dataset with more images per class to further improve accuracy.
Experiment with larger architectures such as EfficientNetB4 or Vision Transformers.
Add Grad-CAM visualisations to highlight prediction-influencing image regions.
Deploy the app to a cloud platform (Hugging Face Spaces or Streamlit Cloud).
