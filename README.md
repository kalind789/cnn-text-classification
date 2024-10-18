# CNN for Sentence Classification

This project implements a Convolutional Neural Network (CNN) for text classification based on the paper **"Convolutional Neural Networks for Sentence Classification"**. The goal is to build a deep learning model that classifies text data, such as sentiment analysis or categorizing text into different topics.

## Project Overview
- **Dataset**: The project uses text datasets such as the IMDb Movie Reviews dataset for training and testing.
- **Model Architecture**: The CNN model uses Word2Vec embeddings to convert text into dense vectors and processes them through convolutional and pooling layers to extract important features for classification.
- **Tools**: 
  - **PyTorch**: For building and training the neural network.
  - **NLTK**: For text preprocessing (tokenization, stopword removal).
  - **Gensim**: For loading pre-trained Word2Vec embeddings.

## Key Features
- Implements custom tokenization and padding for uniform input size.
- Uses the PyTorch DataLoader for efficient data loading and batching.
- Supports dynamic sequence padding using a custom collate function.

## How to Run
1. Set up the environment using the provided `environment.yml` file.
2. Preprocess the text data using the functions in the `notebooks` folder.
3. Train the model using the PyTorch implementation in `src/model.py`.

## Future Improvements
- Experimenting with different embedding techniques like GloVe or BERT.
- Extending the model to handle multi-class classification tasks.

## References
- **Paper**: Yoon Kim, *Convolutional Neural Networks for Sentence Classification*. [Link](https://arxiv.org/abs/1408.5882)
