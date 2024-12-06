# diagnose.py

import math
import numpy as np
import json

# Helper function to calculate state index
def get_state_index(asked_symptoms):
    return int("".join(map(str, asked_symptoms.astype(int))), 2)

def diagnose_patient(q_table, symptoms, disease_symptom_prob, idx_to_disease, n_diseases, max_questions):
    n_symptoms = len(symptoms)
    symptoms_state = np.zeros(n_symptoms, dtype=int)  # Initial symptom state (all unknown)
    asked_symptoms = np.zeros(n_symptoms, dtype=int)  # Track asked symptoms
    

    for ques_num in range(max_questions):
        state_idx = get_state_index(asked_symptoms)

        # Choose action based on Q-table
        q_values = q_table[state_idx].copy()
        # Exclude already asked symptoms from possible actions
        q_values[np.where(asked_symptoms == 1)[0]] = -np.inf
        action = np.argmax(q_values)
        print("Action:", action)
        print("Asked Symptoms:", ques_num + 1)

        if action < n_symptoms:  # Ask about a symptom
            if asked_symptoms[action] == 1:
                continue  # Already asked, skip
            asked_symptoms[action] = 1
            # Ask the patient
            while True:
                try:
                    patient_response = int(input(f"Do you have symptom '{symptoms[action]}'? (1 for Yes, 0 for No): "))
                    if patient_response in [0, 1]:
                        break
                    else:
                        print("Please enter 1 for Yes or 0 for No.")
                except ValueError:
                    print("Invalid input. Please enter 1 or 0.")
            symptoms_state[action] = patient_response

            # Compute disease probabilities after each symptom inquiry
            observed_symptoms = np.where(asked_symptoms == 1)[0]
            disease_log_likelihoods = []
            for disease_idx in range(n_diseases):
                disease_name = idx_to_disease[str(disease_idx)]
                epsilon_prob = 1e-6
                log_likelihood = 0
                for symptom_idx in observed_symptoms:
                    prob = disease_symptom_prob[disease_name][symptom_idx]
                    prob = min(max(prob, epsilon_prob), 1 - epsilon_prob)
                    if symptoms_state[symptom_idx] == 1:
                        log_likelihood += math.log(prob)
                    else:
                        log_likelihood += math.log(1 - prob)
                disease_log_likelihoods.append(log_likelihood)

            # Convert log-likelihoods to probabilities
            max_log_likelihood = max(disease_log_likelihoods)
            log_likelihoods_shifted = [ll - max_log_likelihood for ll in disease_log_likelihoods]
            exp_likelihoods = [math.exp(ll) for ll in log_likelihoods_shifted]
            total_likelihood = sum(exp_likelihoods)
            disease_probabilities = [exp_ll / total_likelihood for exp_ll in exp_likelihoods]

            # Get top 3 diseases by probability
            top_3_indices = np.argsort(disease_probabilities)[-3:][::-1]
            top_3_diseases = [(idx_to_disease[str(i)], disease_probabilities[i]) for i in top_3_indices]

            print("\nTop 3 Diseases:")
            for disease, prob in top_3_diseases:
                print(f"{disease}: Probability({prob:.2f})")

            # Check if the most likely disease probability exceeds 0.9
            if top_3_diseases[0][1] >= 0.9:
                print(f"\nConfident Diagnosis: {top_3_diseases[0][0]} (Probability: {top_3_diseases[0][1]:.2f})")
                return top_3_diseases


        else:  # Make a prediction (if max_questions reached or policy dictates)
            # Predict disease based on observed symptoms
            observed_symptoms = np.where(asked_symptoms == 1)[0]
            disease_log_likelihoods = []
            for disease_idx in range(n_diseases):
                disease_name = idx_to_disease[str(disease_idx)]
                epsilon_prob = 1e-6
                log_likelihood = 0
                for symptom_idx in observed_symptoms:
                    prob = disease_symptom_prob[disease_name][symptom_idx]
                    prob = min(max(prob, epsilon_prob), 1 - epsilon_prob)
                    if symptoms_state[symptom_idx] == 1:
                        log_likelihood += math.log(prob)
                    else:
                        log_likelihood += math.log(1 - prob)
                disease_log_likelihoods.append(log_likelihood)

            # Convert log-likelihoods to probabilities
            max_log_likelihood = max(disease_log_likelihoods)
            log_likelihoods_shifted = [ll - max_log_likelihood for ll in disease_log_likelihoods]
            exp_likelihoods = [math.exp(ll) for ll in log_likelihoods_shifted]
            total_likelihood = sum(exp_likelihoods)
            disease_probabilities = [exp_ll / total_likelihood for exp_ll in exp_likelihoods]

            top_3_indices = np.argsort(disease_probabilities)[-3:][::-1]
            top_3_diseases = [(idx_to_disease[str(i)], disease_probabilities[i]) for i in top_3_indices]

            print("\nFinal Top 3 Diseases:")
            for disease, prob in top_3_diseases:
                print(f"{disease}: Probability ({prob:.2f})")
            return top_3_diseases

    print("Max questions reached. Unable to make a confident diagnosis.")
    return None

# Load pre-trained Q-table
q_table = np.load('q_table.npy')

# Load disease symptom probabilities
with open('disease_symptom_prob.json', 'r') as f:
    disease_symptom_prob = json.load(f)

with open('idx_to_disease.json', 'r') as f:
    idx_to_disease = json.load(f)

with open('symptoms.json', 'r') as f:
    symptoms = json.load(f)

n_diseases = len(idx_to_disease)

# Diagnose a new patient
predicted_disease = diagnose_patient(q_table, symptoms, disease_symptom_prob, idx_to_disease, n_diseases, len(symptoms))
