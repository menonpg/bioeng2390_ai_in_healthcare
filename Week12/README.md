# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. This course includes comprehensive introduction to programming in Python, R, and MATLAB, as well as setting up development environments and using Integrated Development Environments (IDEs) to develop linear, nonlinear, and deep learning models for healthcare applications, starting with time series data, image data, and text data.


# Week 12 Lecture 23: Transfer Learning and Image Classification

## Course Updates
- We're approaching the end of the semester
- Schedule for remaining classes:
  - Week 12 (today) - April 8, April 10
  - Week 13 - April 15, April 17 (no class)
  - Final project presentations - April 22
  - No class on April 24
- All final assignments and project materials due by April 24

## Neural Networks in PyTorch and TensorFlow

### MLP from Scratch (PyTorch)
- Review of the MLP model built in the previous class for RGB color classification
- Model architecture:
  - Input layer (3 nodes for RGB values)
  - Hidden layer (configurable number of neurons)
  - Output layer (1 node for binary classification)
- Key components:
  - Data normalization (RGB values divided by 255)
  - PyTorch tensors for data representation
  - Binary cross-entropy loss with sigmoid activation
  - Adam optimizer for weight updates
- Training process:
  - Configurable epochs and batch size
  - Loss decreases over training iterations
  - Model can be saved for later use

### Equivalent Model in TensorFlow/Keras
- Same functionality implemented using TensorFlow/Keras API
- Key differences in implementation:
  - Sequential model definition instead of class-based approach
  - Layer-by-layer construction using Dense layers
  - Built-in model compilation and fitting methods
  - Similar training process with epochs and batches

## Image Classification with Teachable Machine

### Mask Detection Dataset
- Dataset contains images of people with and without masks
- Images include variations (rotated, translated, scaled, blurred)
- Data augmentation creates robust models by exposing them to expected variations

### Using Teachable Machine
1. Upload labeled images (with mask/without mask)
2. Train the model with minimal configuration
3. Test the model using webcam or uploaded images
4. Export the model (TensorFlow.js, Python, etc.)

### Loading and Using the Exported Model
- Model is saved as an H5 file with labels
- Can be loaded and used for inference in Python
- Architecture typically includes a large feature extraction layer and output classification layer

## Transfer Learning

### Concept of Transfer Learning
- Using pre-trained models as feature extractors
- Leveraging knowledge from models trained on massive datasets
- Examples: InceptionV3, MobileNet (trained on ImageNet with 1000 classes)

### Image Embeddings
- Converting images into feature vectors (embeddings)
- These embeddings capture semantic understanding of the image
- Benefits:
  - Better feature representation than raw pixels
  - Can be used for similarity search
  - Enables transfer learning

### Transfer Learning Implementation
1. Load a pre-trained model (InceptionV3, MobileNet)
2. Remove the classification layers
3. Add new trainable layers:
   - Flatten layer
   - Dense hidden layer (e.g., 1024 neurons)
   - Dropout layer (e.g., 20%)
   - Output layer with appropriate activation
4. Freeze the weights of the pre-trained layers
5. Train only the new layers

### Data Augmentation
- Enhances model robustness by creating variations
- Techniques include rotation, translation, scaling, etc.
- Can be integrated into the training pipeline

### GPU Acceleration
- Training deep learning models is much faster on GPUs
- Google Colab provides free GPU access
- Minor code modifications needed to utilize GPU resources

## Key Metrics for Model Evaluation
- Classification metrics review:
  - Sensitivity/Recall: TP/(TP+FN)
  - Specificity: TN/(TN+FP)
  - Precision: TP/(TP+FP)
  - F1 Score: 2(Precision×Recall)/(Precision+Recall)
- Different metrics are appropriate for different use cases
- F1 score helps find a balance that favors precision over sensitivity

## Next Steps
- Running the transfer learning model on GPU
- Completing the training process
- Evaluating model performance
- Applying these techniques to your own projects