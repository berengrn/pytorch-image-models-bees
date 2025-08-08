import torch
import torch.nn as nn
from timm.utils.metrics import *
from timm.loss.hierarchical_jsd import Build_H_Matrix,BuildDictionaries

class HierarchicalAccuracy(nn.Module):
    def __init__(self,csv_file):
        self.csv_file = csv_file
        _,_,self.piquets = BuildDictionaries(csv_file)
        self.nbLevels = len(self.piquets) - 1

    def compute(self,output,target,topk=1):
        """
        params: output, target: must be of size batch_size x total_classes
        """
        result = torch.tensor(0.0)
        for l in range(self.nbLevels):
            k = 0 #Compteur de niveaux pour lesquels on a calculé une accuracy
            if max(topk) < self.piquets[l+1] - self.piquets[l]:
                k+=1
                result += accuracy(output[:,self.piquets[l]:self.piquets[l+1]],
                                torch.argmax(target[:,self.piquets[l]:self.piquets[l+1]], dim=1), topk=topk)[0]
                result /= k
        return result
    


