# Twitter Sentiment Analysis using DistilBERT

## Overview

This project is focused on developing a **sentiment analysis model** using **DistilBERT**, a smaller, faster variant of the BERT (Bidirectional Encoder Representations from Transformers) model. The primary goal is to fine-tune the pre-trained DistilBERT model on a custom Twitter dataset, which contains various tweets with labeled sentiments. Sentiment analysis is crucial for understanding user opinions, emotions, and trends in social media.

## Objectives

- Fine-tune a **pre-trained DistilBERT** model using a custom Twitter dataset.
- Classify tweets into various sentiment categories, enhancing the model’s understanding of tweet contexts and sentiments.
- Deploy the model efficiently on **GPU** to leverage faster training and evaluation.
- Provide insights into the **data preparation**, **model training**, and **evaluation** processes to showcase how DistilBERT can be effectively fine-tuned for sentiment analysis.

## Dataset Description

The dataset comprises tweets extracted from **Twitter**. Each tweet in the dataset has a corresponding sentiment label, typically reflecting categories such as:

- Positive
- Negative
- Neutral
- Irrelevant

This labeled dataset is used to fine-tune the DistilBERT model for performing accurate sentiment classification.

## Approach

### 1. Data Preprocessing

Before fine-tuning the DistilBERT model, the dataset undergoes preprocessing, including:

- **Cleaning**: Removing unnecessary characters, symbols, and stop words from the tweets.
- **Tokenization**: Tokenizing the input text using the **DistilBERT tokenizer** to ensure that the input format is compatible with the pre-trained model.
- **Label Encoding**: Encoding the sentiment labels into a numerical format to be used during training.

### 2. Model Fine-tuning

- **Pre-trained Model**: The model used is **DistilBERT**, a lightweight version of BERT that retains most of BERT's accuracy but is faster and more efficient.
- **Fine-tuning**: The pre-trained DistilBERT model is fine-tuned on the Twitter dataset by adjusting the weights through backpropagation using the dataset’s sentiment labels.

### 3. GPU Utilization

- The model training is optimized for **GPU** to significantly reduce the computational time.
- **PyTorch** and the Hugging Face **Transformers** library are used to handle GPU-based model training, ensuring fast and efficient processing.

### 4. Training Process

- The dataset is split into **training** and **validation** sets and trained accordingly.
  
### 5. Evaluation Metrics

- The performance of the model is evaluated using:
  - **Accuracy**: The percentage of correctly classified sentiments.
  - **F1 Score**: A weighted average of precision and recall, offering a balance between both metrics.
  - **Precision**: A unit that gives the score based on true positives and total positives predicted i.e., true positives and false positives.
  - **Recall**: The ratio between true positives and number of positives in the given data set(true positives and false negatives).

### 6. Predictions

Once fine-tuned, the model can predict the sentiment of unseen tweets, providing accurate sentiment analysis results in real-time.

## Key Features

- **Lightweight**: DistilBERT is faster and requires fewer resources compared to BERT, making it ideal for projects with limited computational resources.
- **GPU-Accelerated**: The entire training process leverages GPU acceleration, significantly speeding up the model fine-tuning.
- **High Accuracy**: The fine-tuned model offers high accuracy in predicting tweet sentiments, making it suitable for real-world applications.
- **Scalable**: The model can be extended to analyze larger datasets or be integrated into applications requiring sentiment analysis.

## Installation and Requirements

To reproduce the results or run the code, the following packages and dependencies are required:

- **Python 3.7+**
- **PyTorch** (with GPU support for efficient training)
- **Transformers** library from Hugging Face
- **Pandas** for data manipulation
- **Sklearn** for metrics and preprocessing

## Future Enhancements

- **Hyperparameter Tuning**: Further improvements could be achieved by experimenting with different hyperparameters like learning rate, batch size, and optimizer types.
- **Advanced Preprocessing**: Implementing techniques such as lemmatization and word embeddings to enhance input text quality.
- **Integration**: Deploying the model as a REST API for real-time sentiment analysis on new tweets.

## Conclusion

This project demonstrates how **DistilBERT** can be fine-tuned for sentiment analysis tasks using a custom Twitter dataset. The efficient GPU usage and model’s adaptability make it an excellent choice for sentiment analysis on social media data, with promising results and room for further enhancements.
