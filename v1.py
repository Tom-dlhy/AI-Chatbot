'''
Chatbot using bag of words with simple Pytorch Neural Network
'''


import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fct1 = nn.Linear(input_size, 128)
        self.fct2 = nn.Linear(128, 64)
        self.fct3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fct1(x))
        x = self.dropout(x)
        x = self.relu(self.fct2(x))
        x = self.dropout(x)
        x = self.fct3(x)

        return x
    
class ChatbotAssistant:
    def __init__(self, intents_path):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_response = {}

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmmatizer.lemmatize(word.lower()) for word in words]

        return words
    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as file:
                intents_data = json.load(file)


            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_response[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags) # self.X = torch.tensor(bags, dtype=torch.float32)
        self.y = np.array(indices) # self.y = torch.tensor(indices, dtype=torch.long)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward() # backpropagation step to compute the gradients
                optimizer.step() # update the weights based on the gradients
                running_loss += loss 

            print(f'Epoch {epoch + 1}, loss: {running_loss/ len(loader):.4f}')

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as file:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents),
            }, file)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as file:
            dimensions = json.load(file)


        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(bag_tensor)

        predicted_class_index = torch.argmax(prediction, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.intents_response[predicted_intent]:
            response = random.choice(self.intents_response[predicted_intent])
        else:
            return None
        
        return response
        

if __name__ == '__main__':
    # assistant = ChatbotAssistant('intents.json')
    # assistant.parse_intents()
    # assistant.prepare_data()
    # assistant.train_model(batch_size=8, lr=0.001, epochs=200)
    # assistant.save_model('chatbot.pth', 'dimensions.json')

    assistant = ChatbotAssistant('intents.json')
    assistant.parse_intents()
    assistant.load_model('chatbot.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')
        if message.lower() == 'exit':
            break
        print(f'User: {message}', flush=True)
        response = assistant.process_message(message)
        print(f'Bot: {response}', flush=True)

