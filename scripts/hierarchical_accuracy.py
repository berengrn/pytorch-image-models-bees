import torch
import torch.nn as nn
from timm.utils.metrics import *
from timm.loss.hierarchical_jsd import Build_H_Matrix,BuildDictionaries

class HierarchicalAccuracy(nn.Module):
    def __init__(self,csv_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_file = csv_file
        parent_to_children,_,self.piquets = BuildDictionaries(csv_file)
        self.H = Build_H_Matrix(parent_to_children,self.piquets).to(self.device)
        self.nbLevels = len(self.piquets) - 1

    def compute(self,output,target,topk=1):
        """
        params: output, target: must be of size batch_size x total_classes
        """
        levels_target = [target[:,-self.piquets[-1]:]]
        for l in range(self.nbLevels - 1, 0, -1):
            level = (self.H[self.piquets[l-1]:self.piquets[l],self.piquets[l]:self.piquets[l+1]] @ levels_target[0].t()).t()
            levels_target.insert(0,level)

        result = torch.tensor(0.0,device=self.device)
        k = 0 #Compteur de niveaux pour lesquels on a calculé une accuracy (certains niveaux n'ont pas assez de classes pour acc5)
        for l in range(self.nbLevels):
            if max(topk) < self.piquets[l+1] - self.piquets[l]:
                k+=1
                result += accuracy(output[:,self.piquets[l]:self.piquets[l+1]],
                                torch.argmax(levels_target[l],dim=1), topk=topk)[0].to(self.device)
        result /= k
        return result