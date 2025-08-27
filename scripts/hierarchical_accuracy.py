import torch
import torch.nn as nn
from timm.utils.metrics import *
from timm.loss.hierarchical_jsd import Build_H_Matrix,BuildDictionaries

class HierarchicalAccuracy(nn.Module):
    def __init__(self,csv_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_file = csv_file
        _,_,self.piquets = BuildDictionaries(csv_file)
        self.nbLevels = len(self.piquets) - 1

    def compute(self,output,target,topk=1):
        """
        params: output, target: must be of size batch_size x total_classes
        """
        result = torch.tensor(0.0,device=self.device)
        k = 0 #Compteur de niveaux pour lesquels on a calculé une accuracy
        for l in range(self.nbLevels):
            if max(topk) < self.piquets[l+1] - self.piquets[l]:
                k+=1
                result += accuracy(output[:,self.piquets[l]:self.piquets[l+1]],
                                torch.argmax(target[:,self.piquets[l]:self.piquets[l+1]], dim=1), topk=topk)[0].to(self.device)
        result /= k
        return result
    


