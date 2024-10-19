# CNN for Sentence Classification

This project implements a Convolutional Neural Network (CNN) for text classification based on the paper **"Convolutional Neural Networks for Sentence Classification"** by Yoon Kim. The goal is to build a deep learning model that classifies text data into categories, such as sentiment analysis or topic categorization, using a vocabulary-based approach with custom embeddings.

## Project Overview
- **Dataset**: The project uses text datasets, such as the IMDb Movie Reviews dataset, for training and testing purposes. The dataset is preprocessed to convert text into sequences of word indices based on a custom vocabulary.
- **Model Architecture**: The CNN model includes:
  - An **embedding layer** that maps word indices to dense vectors.
  - Multiple **convolutional layers** with varying filter sizes (e.g., 2, 3, and 4) to capture different n-grams and patterns in the text.
  - **Max-pooling layers** to extract the most significant features from each filter.
  - A **fully connected layer** that combines these features for final classification.
- **Tools**: 
  - **PyTorch**: For building and training the neural network using the custom embeddings and CNN architecture.
  - **NLTK**: For text preprocessing, including tokenization and stopword removal.
  - **Pandas**: For handling the dataset and managing data structures.
  - **NumPy**: For efficient numerical operations during preprocessing.

## Project Implementation
1. **Data Preprocessing**:
   - Text data is tokenized using `NLTK`, and a custom vocabulary is created.
   - Tokens are converted into numerical indices based on the vocabulary and padded to a fixed length.
2. **Model Training**:
   - The CNN is trained using the padded sequences with a cross-entropy loss function and the Adam optimizer.
   - Training includes multiple epochs with monitoring of loss and accuracy to evaluate performance.
3. **Evaluation**:
   - After training, the model is evaluated on a test dataset to measure accuracy and generalization capability.

## Future Improvements
- Experimenting with different embedding techniques, such as **GloVe** or fine-tuning **BERT** embeddings, to improve model performance.
- Extending the model to handle **multi-class classification** tasks for more complex text datasets.
- Implementing **dropout** and **batch normalization** for improved regularization and training stability.

## How to Run
1. Install the required dependencies listed in `requirements.txt`.
2. Preprocess the dataset using the scripts provided in the `preprocessing` folder.
3. Train the model using `train.py`, specifying any necessary hyperparameters (e.g., learning rate, batch size).
4. Evaluate the model on a test dataset using `evaluate.py`.

## References
- **Paper**: Yoon Kim, *Convolutional Neural Networks for Sentence Classification*. [Link](https://arxiv.org/abs/1408.5882)

## Additional Notes
- This implementation uses a vocabulary approach where embeddings are learned from scratch during training.
- For efficient training, using Google Colab with GPU or setting up a cloud environment (AWS, Azure, GCP) with GPU support is recommended.
