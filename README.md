# CNN for Sentence Classification

This project implements a Convolutional Neural Network (CNN) for text classification based on the paper **"Convolutional Neural Networks for Sentence Classification"** by Yoon Kim. The notebook covers the entire workflow, from data preprocessing to model training and evaluation.

## Project Overview
- **Notebook**: The entire implementation, including preprocessing, model building, training, and evaluation, is contained within a single Jupyter Notebook (`cnn-text-classification.ipynb`).
- **Dataset**: The project uses text datasets such as the IMDb Movie Reviews dataset for training and testing. The dataset is preprocessed to convert text into sequences of word indices based on a custom vocabulary.
- **Model Architecture**: 
  - The CNN model includes an embedding layer that maps word indices to dense vectors.
  - Multiple convolutional layers with varying filter sizes (e.g., 2, 3, and 4) capture different n-grams and patterns in the text.
  - Max-pooling layers extract the most significant features from each filter output.
  - A fully connected layer combines these features for final classification.
- **Tools**:
  - **PyTorch**: For building and training the neural network using the custom embeddings and CNN architecture.
  - **NLTK**: For text preprocessing (e.g., tokenization, stopword removal).
  - **Pandas** and **NumPy**: For handling the dataset and managing data structures.

## Running the Notebook
1. **Install Dependencies**:
   - Make sure you have the required libraries installed. If using `conda`, you can set up the environment:
     ```bash
     conda create -n cnn-text python=3.9
     conda activate cnn-text
     pip install torch nltk pandas numpy jupyter
     ```
2. **Run the Notebook**:
   - Open the notebook:
     ```bash
     jupyter notebook cnn-text-classification.ipynb
     ```
   - Follow the steps in the notebook to preprocess the data, train the model, and evaluate it on the test dataset.

## Future Improvements
- Experimenting with different embedding techniques, such as **GloVe** or fine-tuning **BERT** embeddings, to improve model performance.
- Extending the model to handle **multi-class classification** tasks for more complex text datasets.
- Implementing additional regularization techniques like **dropout** and **batch normalization** for improved training stability.

## References
- **Paper**: Yoon Kim, *Convolutional Neural Networks for Sentence Classification*. [Link](https://arxiv.org/abs/1408.5882)

## Notes
- This implementation uses a vocabulary approach where embeddings are learned from scratch during training.
- For efficient training, using Google Colab with GPU or setting up a cloud environment (AWS, Azure, GCP) with GPU support is recommended.
