import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv

from collections import defaultdict

def BuildDictionaries(csv_path):

    data = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        nbLevels = len(header)
        levels = [[] for _ in range(nbLevels)]

        for row in reader:
            for i in range(nbLevels):
                if row[i] not in levels[i]:
                    levels[i].append(row[i])
        #print("levels: ",levels)
        
        csvfile.seek(0)
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data.append([row[i] for i in range(nbLevels)])

        #print("data: ",data)

        #print("levels shape:",len(levels),"x",len(levels[-1]))
        #print("data shape:",len(data),len(data[0]))
   
    # Calcule l'offset pour chaque niveau
    offsets = np.cumsum(np.array([0] + [len(l) for l in levels]))
    level_idxs = [{name:i + offsets[k] for i, name in enumerate(levels[k])} for k in range(nbLevels)]
    idxs = {}
    for l in level_idxs:
        idxs |= l
    #print("level_idxs: ",level_idxs,"\n\nidxs: ",idxs)

    # Fusion pour accès global
    #idx_all = {**idx_ordres, **idx_familles, **idx_genres, **idx_especes}

    parent_to_children = defaultdict(set)

    for info in data:
        for i in range(nbLevels - 1):
            parent_to_children[idxs[info[i]]].add(idxs[info[i+1]])

    parent_to_children = {k: list(v) for k, v in parent_to_children.items()}
    #print("parent_to_children: ",parent_to_children)

    return (parent_to_children,idxs,offsets)

def Build_H_Matrix(parent_to_children,piquets):
    #print("piquets: ",piquets)
    #H[i][j] == 1 iff i parent of j, else 0
    N = piquets[-1]
    H = torch.zeros((N,N))

    for i in range(N):
        if i in parent_to_children:
            for j in parent_to_children[i]:
                H[i][j] = 1

    return H

class HierarchicalJsd(nn.Module):
    """
    param: csv_path: chemin vers le fichier .csv contenant la hiérarchie des classes
    """


    def __init__(self,csv_path):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parent_to_children,_,self.piquets = BuildDictionaries(csv_path)
        self.H = Build_H_Matrix(parent_to_children,self.piquets).to(device)

    def forward(self,y_pred,target): 
        """
        :param target: Tensor de forme (batch_size,) avec les indices des classes cibles pour le dernier niveau.
        (les niveaux supérieurs seront retrouvés automatiquement a partir de la hiérarchie)
        """
        device = target.device
        nbLevels = len(self.piquets) - 1

        #probabilité d'une classe en fonction des probabilités prédites pour ses enfants:
        children_preds = (self.H[:self.piquets[-2],:] @ y_pred.t()).t()

        #On découpe les vecteurs par niveau hiérarchique
        levels_pred = [y_pred[:,self.piquets[k]:self.piquets[k+1]]  for k in range(nbLevels)]
        levels_children = [children_preds[:,self.piquets[i]:self.piquets[i+1]] for i in range(nbLevels - 1)]

        #logs pour le softmax
        log_levels_pred = [F.log_softmax(level,dim=1) for level in levels_pred]
        log_levels_children = [F.log_softmax(level,dim=1) for level in levels_children]
        
        #targets pour chaque niveau
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
        
        target += self.piquets[-2] #dataset labels concern the lowest hierarchic level
        levels_target = torch.stack([torch.stack ( [torch.tensor(get_grandparents(nbLevels - i,y),device=device) for i in range(1,nbLevels+1)] ) for y in target])
        levels_target = torch.transpose(levels_target,0,1)
        for k in range(nbLevels):
            levels_target[k] -= self.piquets[k]  #indice de la classe correcte dans un niveau
        #levels_target = levels_target.to(device)


        loss,kl_penalty = torch.tensor(0.0,device=device),torch.tensor(0.0,device=device)
        kl_loss = torch.nn.KLDivLoss(reduction='mean')
        ce_fn = nn.CrossEntropyLoss()

        for k in range(nbLevels - 1):
            #turn levels_pred and levels_children into "probability distributions":

            #Divergence de Jensen-Shannon:
            kl_penalty += 0.5*(kl_loss(log_levels_pred[k], F.softmax(levels_children[k],dim=1) ) 
                                + kl_loss(log_levels_children[k], F.softmax(levels_pred[k],dim=1)) )

            loss += 0.5*ce_fn(levels_pred[k],levels_target[k])
        loss += 0.5*ce_fn(levels_pred[-1],levels_target[-1])

        loss += 0.5*kl_penalty
        return loss