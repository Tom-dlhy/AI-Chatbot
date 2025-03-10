'''
Chatbot using SBERT and simple Pytorch Neural Network
'''


import os
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sentence_transformers import SentenceTransformer


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
    def __init__(self, intents_path, sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')):
        self.model = None
        self.intents_path = intents_path
        self.sbert_model = sbert_model

        self.intents_response = {}
        self.intents_embeddings = {}
        self.intents = []

        self.X = None
        self.y = None

    def parse_intents(self):

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as file:
                intents_data = json.load(file)


            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_response[intent['tag']] = intent['responses']

                self.intents_embeddings[intent['tag']] = []

                for pattern in intent['patterns']:
                    embedding_pattern = self.sbert_model.encode(pattern)
                    self.intents_embeddings[intent['tag']].append(embedding_pattern)

            self.intents.sort()
        
        else:
            raise FileNotFoundError(f'{self.intents_path} not found')


    def prepare_data(self):
        
        self.X = []
        self.y = []

        self.intent_to_index = {intent: idx for idx, intent in enumerate(self.intents)}

        for tag, embeddings in self.intents_embeddings.items():
            tag_index = self.intent_to_index[tag]
            for embedding in embeddings:
                self.X.append(embedding)
                self.y.append(tag_index)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)

        # Calculer le nombre d'exemples par classe
        y_np = y_tensor.numpy()
        class_counts = np.bincount(y_np)
        # Pour chaque échantillon, on attribue un poids inversement proportionnel à la fréquence de sa classe
        weights = 1.0 / class_counts
        sample_weights = weights[y_np]
        
        # Créer un WeightedRandomSampler pour obtenir un échantillonnage équilibré
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )

        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        self.model = ChatbotModel(input_size=384, output_size=len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()  # Calcul des gradients
                optimizer.step()  # Mise à jour des poids
                running_loss += loss.item()

            print(f'Epoch {epoch + 1}, loss: {running_loss / len(loader):.4f}')


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
        self.model.load_state_dict(torch.load(model_path))

    def process_message(self, input_message):
        embedding_message = self.sbert_model.encode(input_message)
        embedding_message = torch.from_numpy(embedding_message).float()




        self.model.eval()
        with torch.no_grad():
            prediction = self.model(embedding_message.unsqueeze(0))
            prediction = F.softmax(prediction, dim=1)

        predicted_class_index = torch.argmax(prediction, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if predicted_intent in self.intents_response and self.intents_response[predicted_intent]:
            response = self.intents_response[predicted_intent]
        else:
            response = "Je ne suis pas sûr de comprendre."
        
        return response
        


if __name__ == '__main__':

    # choice = input('Train or Load model? (train/load): ')
    # if choice.lower() == 'train':
    #     assistant = ChatbotAssistant('intents.json')
    #     assistant.parse_intents()
    #     assistant.prepare_data()
    #     assistant.train_model(batch_size=8, lr=0.001, epochs=200)
    #     assistant.save_model('chatbot3.pth', 'dimensions3.json')
    # else:
    #     assistant = ChatbotAssistant('intents2.json')
    #     assistant.parse_intents()
    #     assistant.load_model('chatbot3.pth', 'dimensions3.json')

    # choice2 = input('Would you like to chat with the chatbot? (yes/no): ')
    # if choice2.lower() == 'yes':

    #     while True:
    #         message = input('Enter your message: ')
    #         if message.lower() == 'exit':
    #             break
    #         response = assistant.process_message(message)
    #         print(response, flush=True)

    assistant = ChatbotAssistant('intents2.json')
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=16, lr=0.001, epochs=50)

    while True:
        message = input('Enter your message: ')
        if message.lower() == 'exit':
            break
        print(f'User: {message}', flush=True)
        response = assistant.process_message(message)
        print(f'Bot: {response}', flush=True)
