import numpy as np

def learn(concepts, target):
    # Initialize Specific boundary with the first positive example
    specific_h = concepts[0].copy()
    
    # Initialize General boundary with '?'
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        # If the instance is Positive
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # If specific hypothesis doesn't match instance, generalize it
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    # Update general hypothesis to match the generalization
                    for j in range(len(specific_h)):
                        if general_h[j][x] != '?' and general_h[j][x] != specific_h[x]:
                            general_h[j][x] = '?'
        
        # If the instance is Negative
        if target[i] == "No":
            for x in range(len(specific_h)):
                # If instance matches general hypothesis, specialize it
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    # Clean up General Hypothesis by removing empty lists (all '?')
    indices = [i for i, val in enumerate(general_h) if val == ['?'] * len(specific_h)]
    for i in reversed(indices):
        general_h.pop(i)

    return specific_h, general_h

# --- Example Dataset ---
# Features: [Sky, Temp, Humidity, Wind, Water, Forecast]
data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
])

target = np.array(['Yes', 'Yes', 'No', 'Yes'])

s_final, g_final = learn(data, target)

print(f"Final Specific Hypothesis (S): {s_final}")
print(f"Final General Hypothesis (G): {g_final}")