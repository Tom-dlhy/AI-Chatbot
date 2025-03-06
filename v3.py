'''
Chatbot using SBERT and simple Tensorflow Neural Network
'''
import os
import json
import random
import numpy as np

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_size,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class ChatbotAssistant:
    def __init__(self, intents_path, sbert_model=SentenceTransformer('paraphrase-MiniLM-L6-v2')):
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
        input_size = self.X.shape[1]
        output_size = len(self.intents)
        self.model = build_model(input_size, output_size, lr)
        self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs)

    def save_model(self, model_path, dimensions_path):
        self.model.save(model_path)
        with open(dimensions_path, 'w') as file:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents)
            }, file)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as file:
            dimensions = json.load(file)
        self.model = load_model(model_path)

    def process_message(self, input_message):
        embedding_message = np.array(self.sbert_model.encode(input_message), dtype=np.float32)
        embedding_message = np.expand_dims(embedding_message, axis=0)

        prediction = self.model.predict(embedding_message, verbose=0)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_intent = self.intents[predicted_class_index]
        if predicted_intent in self.intents_response and self.intents_response[predicted_intent]:
            response = random.choice(self.intents_response[predicted_intent])
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
    #     assistant.save_model('chatbot2.h5', 'dimensions2.json')
    # else:
    #     assistant = ChatbotAssistant('intents.json')
    #     assistant.parse_intents()
    #     assistant.load_model('chatbot2.h5', 'dimensions2.json')

    # choice2 = input('Would you like to chat with the chatbot? (yes/no): ')
    # if choice2.lower() == 'yes':
    #     while True:
    #         message = input('Enter your message: ')
    #         if message.lower() == 'exit':
    #             break
    #         print(f'User: {message}', flush=True)
    #         response = assistant.process_message(message)
    #         print(f'Bot: {response}', flush=True)

    assistant = ChatbotAssistant('intents.json')
    assistant.parse_intents()
    assistant.load_model('chatbot2.h5', 'dimensions2.json')
    while True:
        message = input('Enter your message: ')
        if message.lower() == 'exit':
            break
        print(f'User: {message}', flush=True)
        response = assistant.process_message(message)
        print(f'Bot: {response}', flush=True)
