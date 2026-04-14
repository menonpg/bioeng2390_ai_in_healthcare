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



# Lecture Notes - Lecture 24: Transfer Learning and Text Embeddings

## Introduction & Housekeeping
- Two more lectures remaining (today and Tuesday)
- No class next Thursday
- Reminder: All assignments due by Thursday, April 24th

## Transfer Learning with MobileNet (Continued from Previous Lecture)

### Setting up GPU Environment in Colab
- Moved transfer learning code to Colab to leverage GPU capabilities
- Steps to enable GPU:
  1. Runtime → Change runtime type → Select T4 GPU
  2. Upload dataset to Colab environment
  3. Encountered version compatibility issues between local environment and Colab

### Dealing with Version Compatibility Issues
- Attempted to install specific versions of TensorFlow and Keras
- Explored options to backdate Python version to 3.9.16 (from instructor's environment)
- Used generative AI to help upgrade code to work with latest versions of TensorFlow and Keras

### Running Transfer Learning Model with GPU
- Added code to explicitly check for and use GPU:
  ```python
  import tensorflow as tf
  print("Using GPU:", tf.config.list_physical_devices('GPU'))
  ```
- Verified GPU usage through resource monitoring
- Successfully trained the model with higher efficiency using GPU

### Model Results
- Achieved 99% accuracy on mask/no mask classification
- Model saved as H5 file that can be loaded in any compute environment
- Demonstrated prediction on new observations

## Text Embeddings and Semantic Similarity

### Introduction to Text Embeddings
- Similar to how image classification models convert images to numbers, text embedding models convert text to vectors
- Embeddings can be used for semantic similarity comparison between texts
- Demonstrated using OpenAI's text embedding model (text-embedding-ada-002)

### Creating Text Embeddings with OpenAI API
- Used OpenAI API to convert text to 1536-dimensional vectors
- Token limit of 8,191 (approximately 6,000-6,500 words)
- Calculated similarity between texts using cosine similarity

### Comparing Different Embedding Models
- Demonstrated sentence-transformers library with all-MiniLM-L6-v2 model
- This BERT-based model produces 384-dimensional embeddings
- Showed how different models have different performance for semantic similarity tasks
- all-MiniLM-L6-v2 provided better differentiation between similar and dissimilar texts

### Applications of Text Embeddings
- Semantic search and retrieval
- Vector databases for storing embeddings
- Retrieval augmented generation (RAG)
- Similarity scoring as features for classification models
- Example applications:
  - Matching job applications to job descriptions
  - Clinical text analysis using bioclinical BERT models

### Using Hugging Face Models
- Demonstrated how to use models from Hugging Face for embeddings
- Example of bioclinical BERT for healthcare text analysis
- Models can be loaded locally and used for domain-specific embedding tasks

## Next Class Preview
- Will cover additional machine learning applications beyond classification and embeddings
- Topics will include object detection and other advanced model applications

## Project Updates
- Asynchronous meetings available for any issues or questions