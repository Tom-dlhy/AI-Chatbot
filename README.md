# 🤖 ChatBot-Project

Welcome to **ChatBot-Project**! This repository provides three chatbot scripts (v1, v2, and v3) that enable you to build intelligent conversational agents. Each script uses a different paradigm for language processing—**Bag of Words** or **SBERT**—and **PyTorch** or **TensorFlow** for training the neural network.

---

## 📋 Project Overview

The goal of this project is to create a chatbot that can answer questions or engage in natural dialogue, using an **intents** file (`intents.json`). Each script implements its own pipeline:

- **v1.py**: Bag of Words + PyTorch  
- **v2.py**: SBERT (Sentence-BERT) + PyTorch  
- **v3.py**: SBERT (Sentence-BERT) + TensorFlow (Keras)

> **Note**: This project is currently focused on **intent classification**. Future versions aim to introduce **text generation** features using **GPT** or other **Transformers** models.

The `intents.json` file contains the intent categories (tags), sample user “patterns,” and predefined responses.

---

## 🛠️ Technologies & Libraries

- **Python**: 3.7 or higher  
- **PyTorch** (v1 & v2) / **TensorFlow** (v3)  
- **NLTK** (for the Bag of Words version)  
- **Sentence Transformers** (for SBERT)  
- **NumPy**, **json**, **os**, **random**

---

## 🌟 Key Features

1. **Multiple Approaches**: Choose between a Bag of Words setup or SBERT embeddings, depending on your performance/accuracy requirements.  
2. **Neural Network Models**:  
   - **v1** and **v2** use MLPs (Multi-Layer Perceptrons) in **PyTorch**.  
   - **v3** uses an MLP in **TensorFlow (Keras)**.  
3. **Intent Management**: Each script relies on `intents.json`, where each *intent* has *patterns* and *responses*.  
4. **Training & Inference**: You can either train the model on your own data or load a pre-trained model.  
5. **Easy Customization**: Adjust layer sizes, dropout rates, number of epochs, etc., with minimal effort.

---

## 📁 Project Structure

- **v1.py**:  
  *ChatbotAssistant* creates a Bag of Words vocabulary, converts sentences into binary vectors, and trains a PyTorch network to classify intents.
- **v2.py**:  
  Uses *Sentence-BERT* to generate 384-dimensional embeddings, then trains a PyTorch network to classify intents.
- **v3.py**:  
  Similar to v2 (SBERT), but uses **TensorFlow (Keras)** instead of PyTorch.
