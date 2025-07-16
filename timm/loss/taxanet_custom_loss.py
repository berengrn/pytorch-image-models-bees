import torch
import torch.nn as nn
import numpy as np
import os
import csv

from collections import defaultdict

def BuildDictionaries(csv_path):

    ordres = []
    familles = []
    genres = []
    especes = []
    data = {}

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            espece = row['species']
            ordre = row['order']
            famille = row['family']
            genre = row['genus']

            if ordre not in ordres:
                ordres.append(ordre)
            if famille not in familles:
                familles.append(famille)
            if genre not in genres:
                genres.append(genre)
            if espece not in especes:
                especes.append(espece)

            data[espece] = {
                'order': ordre,
                'family': famille,
                'genus': genre,
                'species': espece
            }

    # Calcule l'offset pour chaque niveau
    n_ordres = len(ordres)
    n_familles = len(familles)
    n_genres = len(genres)
    n_especes = len(especes)

    idx_ordres = {name: i for i, name in enumerate(ordres)}
    idx_familles = {name: i + n_ordres for i, name in enumerate(familles)}
    idx_genres = {name: i + n_ordres + n_familles for i, name in enumerate(genres)}
    idx_especes = {name: i + n_ordres + n_familles + n_genres for i, name in enumerate(especes)}

    # Fusion pour accès global
    idx_all = {**idx_ordres, **idx_familles, **idx_genres, **idx_especes}

    parent_to_children = defaultdict(set)

    for sp_name, info in data.items():
        order = info['order']
        family = info['family']
        genus = info['genus']
        species = info['species']

        # Ajoute les relations dans le graphe
        parent_to_children[idx_ordres[order]].add(idx_familles[family])
        parent_to_children[idx_familles[family]].add(idx_genres[genus])
        parent_to_children[idx_genres[genus]].add(idx_especes[species])

    parent_to_children = {k: list(v) for k, v in parent_to_children.items()}

    #les piquets indiquent ou s'arrête chaque partie du vecteur correspondant à une niveau hiérarchique
    piquets = torch.cumsum(torch.tensor([0,n_ordres,n_familles,n_genres,n_especes]),0)

    return (parent_to_children,idx_all,piquets)

def Build_H_Matrix(parent_to_children,piquets):
    #Construit H telle que H[i][j] == 1 ssi i est parent de j, 0 sinon
    N = piquets[-1]
    H = torch.zeros((N,N))
    for i in range(N):
        if i in parent_to_children:
            for j in parent_to_children[i]:
                H[i][j] = 1
    return H

class TaxaNetLoss(nn.Module):

    def __init__(self,csv_path):
        super().__init__()
        parent_to_children,_,self.piquets = BuildDictionaries(csv_path)
        self.H = Build_H_Matrix(parent_to_children,self.piquets)
    
    def forward(self,y_pred,y_true):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.piquets = self.piquets.to(device)
        loss = 0.0
        sum = 0.0
        weights = [0.25, 0.25, 0.15, 0.1] #Les poids sont sans doute à re-tester pour notre dataset (pas les mêmes niveaux taxonomiques)
        batch_size,C_total = y_pred.size()
        nbLevels = len(self.piquets) - 1

        """
        N: Number of elements in the dataset
        C_total: Number of classes, all hierarchical levels combined
        nbLevels: Number of levels in the hierarchy
        """
        
        levels_pred = [y_pred[:,self.piquets[k-1]:self.piquets[k]]  for k in range(1,nbLevels+1)]

        #création des cibles pour la cross entropy loss par niveau: numéro de la classe correcte
        def get_grandparents(n: int,id:int): #return the id of the parent from the n-th level above
            if n==0: return id
            N = self.H.size()[0]
            if n >= len(self.piquets) - 1: 
                print("erreur de get_grandparents")
                exit(1)
            for k in range(n):
                for j in range(N):
                    if self.H[j][id] == 1:
                        id = j
                        break
            return id

        y_true += self.piquets[-2] #dataset labels concern the lowest hierarchic level
        levels_true = torch.stack([torch.stack ( [torch.tensor(get_grandparents(nbLevels - i,y),device=device) for i in range(1,nbLevels+1)] ) for y in y_true])
        levels_true = torch.transpose(levels_true,0,1)
        for k in range(nbLevels):
            levels_true[k] -= self.piquets[k]  #indice de la classe correcte dans un niveau
        levels_true = levels_true.to(device)
        
        ce_fn = nn.CrossEntropyLoss()
        sum += ce_fn(levels_pred[0],levels_true[0])
        for k in range(1,nbLevels):
        #on itère d'abord sur chaque niveau hiérarchique
            for i in range(batch_size):
                sum += (self.H[torch.argmax(levels_pred[k-1][i])][torch.argmax(levels_pred[k][i])] == 0)*np.e
            sum += ce_fn(levels_pred[k],levels_true[k]) 
            sum *= weights[k]
            loss += sum

        return loss

if __name__ == '__main__':
    
    csv_path = os.path.join(os.pardir,"small-collomboles","hierarchy.csv")
    loss_fn = TaxaNetLoss(csv_path)