import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv

from collections import defaultdict
from .jsd import JsdCrossEntropy
from .cross_entropy import SoftTargetCrossEntropy

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
        
        csvfile.seek(0)
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data.append([row[i] for i in range(nbLevels)])
   
    # Calcule l'offset pour chaque niveau
    offsets = np.cumsum(np.array([0] + [len(l) for l in levels]))
    level_idxs = [{name:i + offsets[k] for i, name in enumerate(levels[k])} for k in range(nbLevels)]
    idxs = {}
    for l in level_idxs:
        idxs |= l

    parent_to_children = defaultdict(set)

    for info in data:
        for i in range(nbLevels - 1):
            parent_to_children[idxs[info[i]]].add(idxs[info[i+1]])

    parent_to_children = {k: list(v) for k, v in parent_to_children.items()}

    return (parent_to_children,idxs,offsets)

def Build_H_Matrix(parent_to_children,piquets):
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
    hier_weight: weight of the hierarchical penalty. the weight of the "classic loss" is (1 - hier_weight)
    0.0 makes the loss a KLdivLoss
    
    """
    def __init__(self,csv_path,smoothing=0.1,hier_weight=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parent_to_children,_,self.piquets = BuildDictionaries(csv_path)
        self.piquets = torch.from_numpy(self.piquets).to(self.device)
        self.H = Build_H_Matrix(parent_to_children,self.piquets).to(self.device)
        self.leaf_classes = self.piquets[-1] - self.piquets[-2]
        self.smoothing = smoothing
        self.hier_weight = hier_weight

    def forward(self,y_pred,target): 
        """
        :param: target: DOIT ETRE UNE SOFT TARGET Tensor de forme (batch_size,num_leave_classes)
        (les niveaux supérieurs seront retrouvés automatiquement a partir de la hiérarchie)
        """
        device = self.device
        y_pred, target = y_pred.to(device), target.to(device)
        nbLevels = len(self.piquets) - 1
        epsilon = torch.tensor(1e-7,device=device)

        #probabilité d'une classe en fonction des probabilités prédites pour ses enfants:
        children_preds = (self.H[:self.piquets[-2],:] @ y_pred.t()).t()

        #On découpe les vecteurs par niveau hiérarchique
        levels_pred = [y_pred[:,self.piquets[k]:self.piquets[k+1]]  for k in range(nbLevels)]
        levels_children = [children_preds[:,self.piquets[i]:self.piquets[i+1]] for i in range(nbLevels - 1)]

        #logs pour le softmax
        log_levels_pred = [F.log_softmax(level + epsilon,dim=1) for level in levels_pred]
        log_levels_children = [F.log_softmax(level + epsilon,dim=1) for level in levels_children]
        
        levels_target = [target[:,:self.leaf_classes]]
        for l in range(nbLevels - 1, 0, -1):
            level = (self.H[self.piquets[l-1]:self.piquets[l],self.piquets[l]:self.piquets[l+1]] @ levels_target[0].t()).t()
            F.normalize(level) #dans la pratique les coefficients ne somment pas excatement à 1
            levels_target.insert(0,level)

        loss,kl_penalty = torch.tensor(0.0,device=device),torch.tensor(0.0,device=device)
        kl_loss = torch.nn.KLDivLoss(reduction='mean',log_target=True)
        #classic_loss = torch.nn.KLDivLoss(reduction='mean')
        classic_loss = SoftTargetCrossEntropy()

        for k in range(nbLevels - 1):
            #turn levels_pred and levels_children into "probability distributions":

            #Divergence de Jensen-Shannon:
            kl_penalty += 0.5*(kl_loss(log_levels_pred[k], log_levels_children[k]) 
            + kl_loss(log_levels_children[k], log_levels_pred[k]))

            loss += (1 - self.hier_weight)*classic_loss(log_levels_pred[k],levels_target[k])

        loss += (1 - self.hier_weight)*classic_loss(log_levels_pred[-1],levels_target[-1])
        loss += (self.hier_weight)*kl_penalty

        if torch.isnan(loss):
            exit(1)
        return loss