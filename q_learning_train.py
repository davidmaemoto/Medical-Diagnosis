# q_learning_train.py

import math
import numpy as np
import random
import pandas as pd
from collections import defaultdict
import json

# Load dataset
data = pd.read_csv('medical_diagnosis.csv')
symptoms = data.columns[1:]  # Exclude 'disease' column
diseases = data['disease'].unique()
disease_to_idx = {d: i for i, d in enumerate(diseases)}
idx_to_disease = {i: d for d, i in disease_to_idx.items()}

# Convert diseases to numeric labels
data['disease'] = data['disease'].map(disease_to_idx)

# Initialize parameters
n_symptoms = len(symptoms)
n_diseases = len(diseases)
learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.05
epsilon_decay = 0.995
min_epsilon = 1e-5
episodes = 1000

# Precompute symptom probabilities for each disease
disease_symptom_prob = defaultdict(lambda: np.zeros(n_symptoms))

for disease in diseases:
    subset = data[data['disease'] == disease_to_idx[disease]]
    # Avoid zero probabilities by adding a small epsilon
    symptom_means = subset[symptoms].mean().values
    epsilon_prob = 1e-6
    symptom_means = np.clip(symptom_means, epsilon_prob, 1 - epsilon_prob)
    disease_symptom_prob[disease] = symptom_means

# Helper function to calculate state index
def get_state_index(asked_symptoms):
    return int("".join(map(str, asked_symptoms.astype(int))), 2)

# Reward function
def get_reward(predicted, actual, ask_penalty=-1, correct_reward=100, incorrect_penalty=-100):
    if predicted is not None:
        return correct_reward if predicted == actual else incorrect_penalty
    return ask_penalty

# Initialize Q-table
q_table = np.zeros((2 ** n_symptoms, n_symptoms + 1))  # +1 for prediction action

print("Training Q-learning agent...")

# Q-learning algorithm
for episode in range(episodes):
    # Randomly select a patient example
    patient = data.sample(1).iloc[0]
    actual_disease = patient['disease']
    symptoms_state = np.zeros(n_symptoms, dtype=int)
    asked_symptoms = np.zeros(n_symptoms, dtype=int)

    done = False
    while not done:
        state_idx = get_state_index(asked_symptoms)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            possible_actions = [i for i in range(n_symptoms) if asked_symptoms[i] == 0] + [n_symptoms]
            action = random.choice(possible_actions)
        else:
            # Exclude already asked symptoms from possible actions
            q_values = q_table[state_idx].copy()
            q_values[np.where(asked_symptoms == 1)[0]] = -np.inf  # Exclude already asked symptoms
            action = np.argmax(q_values)

        if action < n_symptoms:  # Ask a question
            asked_symptoms[action] = 1
            symptoms_state[action] = patient[symptoms[action]]
            reward = get_reward(None, actual_disease)
        else:  # Make a prediction
            # Predict disease based on current symptom likelihood
            observed_symptoms = np.where(asked_symptoms == 1)[0]
            disease_likelihoods = []
            for disease_idx in range(n_diseases):
                disease_name = idx_to_disease[disease_idx]
                # Avoid log(0) by adding epsilon
                epsilon_prob = 1e-6
                log_likelihood = 0
                for symptom_idx in observed_symptoms:
                    prob = disease_symptom_prob[disease_name][symptom_idx]
                    prob = np.clip(prob, epsilon_prob, 1 - epsilon_prob)
                    if symptoms_state[symptom_idx] == 1:
                        log_likelihood += math.log(prob)
                    else:
                        log_likelihood += math.log(1 - prob)
                disease_likelihoods.append(log_likelihood)
            predicted_disease = np.argmax(disease_likelihoods)
            reward = get_reward(predicted_disease, actual_disease)
            done = True

        # Update Q-table
        next_state_idx = get_state_index(asked_symptoms)
        q_table[state_idx, action] = q_table[state_idx, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
        )
    print(f"Episode {episode + 1}/{episodes}")
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Save the Q-table and other necessary data
np.save('q_table.npy', q_table)

# Convert disease_symptom_prob to a serializable format
disease_symptom_prob_serializable = {k: v.tolist() for k, v in disease_symptom_prob.items()}
with open('disease_symptom_prob.json', 'w') as f:
    json.dump(disease_symptom_prob_serializable, f)

# Convert indices to strings for JSON serialization
idx_to_disease_str = {str(k): v for k, v in idx_to_disease.items()}
with open('idx_to_disease.json', 'w') as f:
    json.dump(idx_to_disease_str, f)

with open('symptoms.json', 'w') as f:
    json.dump(symptoms.tolist(), f)

print("Training complete. Q-table saved as 'q_table.npy'.")
