
---

# RNN Text Generation

This project explores text generation using Recurrent Neural Networks (RNNs) with a focus on Gated Recurrent Units (GRUs). The primary objective is to predict the next set of characters based on the preceding ones, a task with practical applications in predictive text and creative writing.

## Overview

This project involves the following steps:

1. **Data Preprocessing**
    - Convert lines of text into numerical tensors.
    - Create a TensorFlow dataset.
    - Prepare data for training and testing.

2. **Model Definition**
    - Define a GRU-based language model.

3. **Training**
    - Train the model using TensorFlow.

4. **Evaluation**
    - Compute model accuracy using perplexity.
    - Evaluate the model using deep nets.

5. **Text Generation**
    - Generate text using the trained model.

## Table of Contents

1. Data Preprocessing Overview
    - Loading the Data
    - Creating the Vocabulary
    - Converting Lines to Tensors
    - Preparing Data for Training and Testing
    - Creating the TensorFlow Dataset
    - Creating Input and Output for the Model

2. Model Definition
    - Defining the GRU Language Model (GRULM)

3. Training
    - Compiling the Model

4. Evaluation
    - Evaluating Using Deep Nets
    - Calculating Log Perplexity

5. Text Generation
    - Generating Language with the Model

6. Usage

7. Reference

8. License



## 1. Data Preprocessing

### Loading the Data

The dataset is loaded with each sentence structured as one line. Extra spaces are removed, and the cleaned lines are stored in a list.

### Creating the Vocabulary

All lines are concatenated into a single continuous text. Unique characters are identified and collected to form the vocabulary. Special characters for padding and unknown tokens are also introduced.

### Converting Lines to Tensors

Each character in the text is converted into a numerical representation using `tf.strings.unicode_split` and `tf.keras.layers.StringLookup`.

### Preparing Data for Training and Testing

The dataset is split into training and validation sets, with the majority of lines used for training and a small subset reserved for validation.

### Creating the TensorFlow Dataset

The TensorFlow dataset is configured to produce batches of text fragments for training. The dataset generates sequences of a specified length, which are used as inputs for the model.

### Creating Input and Output for the Model

The model predicts the next character in a sequence. Input and output sequences are created by splitting each sequence into two parts: the first part excluding the last character (input) and the second part excluding the first character (output).

## 2. Model Definition

### Defining the GRU Language Model (GRULM)

The GRU model processes character embeddings and makes sequential predictions. The model's architecture includes an embedding layer, GRU layer, and a dense layer with a linear layer and log-softmax computation.

## 3. Training

### Compiling the Model

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. It is trained on the training dataset, and the performance is evaluated on the validation dataset.

## 4. Evaluation

### Evaluating Using Deep Nets

The model's accuracy is measured using perplexity. Lower perplexity indicates better performance.

## 5. Text Generation

### Generating Language with the Model

The trained model generates text by predicting the next character based on preceding characters. The generated text is evaluated for coherence and relevance.



## 6. Usage

To run this project, you need to:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shreyash-Gaur/RNN_Text-Generation.git
   cd RNN_Text-Generation
   ```

2. **Install Dependencies**:
   ```bash
   pip install numpy pandas jupyter tensorflow shutil os
   ```

3. **Run the Jupyter Notebook**:
   Open `Text_Gen.ipynb` in Jupyter Notebook or Jupyter Lab to explore the code, run the cells, train the model, and generate text.

## 7. References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [GRU in TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)

## 8. License

This project is licensed under the MIT License.

---