import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

# Code en python d'un neurone à deux entrées

global nb_entry
nb_entry = 2
def gen_data():
    """
        Génération de la data base (x) et de la target (d)
    """
    if random.uniform(0,1) > 0.5: # Sick : d = 1
        x = [(random.uniform(0,1)-2),(random.uniform(0,1)-2)]
        d = 1
    else:
        x = [(random.uniform(0,1)+2),(random.uniform(0,1)+2)]
        d = 0
    return x, d

def init_variables():
    """
        Génération des poids (w) et du biais (b)
    """
    w = [0] * (nb_entry) # Le tableau avec les poids
    bias = random.uniform(0,1) # On génère le biais aléatoirement
    for loop in range(nb_entry):
        w[loop] = random.uniform(0,1) #On génère les poids aléatoirement
    return w, bias

def pre_activation(x, w, bias):
    """
        Somme pondérée (y) des entrées, poids et du biais
    """
    y = 0 # La variable avec la somme
    for loop in range(nb_entry):
        y += w[loop] * x[loop] # On ajoute wi*xi
    y += bias # On ajoute le biais
    return y

def activation(y):
    """
        Sortie (z), fonction sigmoide
    """
    z = 1 / (1 + np.exp(-y)) # Fonction sigmoide : f(x) = 1 / (1 + e^-x)
    return z

def d_activation(y):
    """
        Dérivation de la fonction sigmoide pour la phase d'apprentissage
    """
    return activation(y) * (1 - activation(y))

def predict(x, w, bias):
    """
        Fonction de prédiction
    """
    y = pre_activation(x, w, bias) # Calcul de y
    z = activation(y) # Calcul de z
    prediction = np.round(z) # On arrondi z pour avoir soit 1, soit 0
    return prediction

def cost_func(z,d):
    """
        Fonction pour définir l'erreur
    """
    cost = 0.5*(z-d)**2
    return cost

def learning(w,bias,nb_batchs=10,nb_epochs=100):
    """
        Phase d'apprentissage (les maths c'est pas mon fort)
    """
    learning_rate = 0.1 # Le learning rate
    x_plt = np.arange(0, nb_batchs*nb_epochs, 1) # Les x d'un plot pour suivre l'erreur au cours de l'apprentissage
    cost_plt = [0] * (nb_batchs * nb_epochs) # Tableau des y avec l'évolution de l'erreur
    i = 0
    # Set des gradients
    w_gradient = [0] * nb_entry
    bias_gradient = 0
    for loop in range(nb_batchs):
        accuracy = [0] * nb_epochs # Historique de l'erreur sur 1 batch
        for loop in range(nb_epochs):
            x, d = gen_data() # On génère les entrées et la sortie voulue
            y = pre_activation(x, w, bias) # On calcule la somme
            z = activation(y) # On calcule la sortie
            accuracy[loop] = (np.round(z)==d)
            cost = cost_func(z, d)
            cost_plt[i] = cost # On prend l'erreur actuelle pour le plot
            i += 1
            cost_history = cost # Mise à jour de l'historique
            # Mise à jour des gradients
            for loop in range(nb_entry):
                w_gradient[loop] += (z - d) * d_activation(y) * x[loop]
            bias_gradient = (z - d) * d_activation(y)
            # Mise à jour des poids et du biais
            for loop in range(nb_entry):
                w[loop] -= w_gradient[loop] * learning_rate
            bias -= bias_gradient * learning_rate
        print(np.mean(accuracy))
    plt.plot(x_plt, cost_plt, 'r')
    plt.show() # On affiche le plot avec l'évolution de l'erreur sur le nombre d'epochs

def save():
    """
        Permet de sauvegarder ses poids et le biais
    """
    with open('save_IA.pickle', 'wb') as f:
        pickle.dump([w,bias], f)

def import_save():
    """
        Permet d'importer les poids et le biais sauvegardés
    """
    with open('save_IA.pickle', 'rb') as f:
        w, bias = pickle.load(f)
    return w, bias

if __name__ == '__main__':
    x, d = gen_data() # On génère les entrées (x) et la sortie attendue (d)
    w, bias = init_variables() # On initie les poids (w) et le biais (b)
    #learning() # On entraine le neurone
