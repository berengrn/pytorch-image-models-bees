import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from pathlib import Path
from timm.loss.hierarchical_jsd import HierarchicalJsd

hierarchy_csv = os.path.join(os.getcwd(),"hierarchy_test.csv")

"""
y_true = torch.tensor([[0,2,5,9],
                       [0,2,6,10],
                       [0,3,7,11]])
                       """

y_true = F.normalize(torch.tensor([[0.5,0.5,0.,0.], #résultat parfait
                          [0.,5.,0.,0.],  
                          [0.,0.,5.,0.]
                        ]),dim=1)

y_true += 1e-7

y_correct = torch.tensor([[5.,0.,5.,0.,0.,5.,0.,0.,0.,5.,0.,0.,0.], #résultat parfait
                          [5.,0.,5.,0.,0.,0.,5.,0.,0.,0.,5.,0.,0.],  
                          [5.,0.,0.,5.,0.,0.,0.,5.,0.,0.,0.,5.,0.]
                        ])

y_false1 = torch.tensor([[5.,0.,0.,0.,5.,5.,0.,0.,0.,5.,0.,0.,0.], #graves erreurs hiérarchiques
                         [5.,0.,5.,0.,0.,0.,0.,5.,0.,0.,5.,0.,0.],  
                         [5.,0.,0.,5.,0.,0.,0.,5.,0.,0.,0.,5.,0.]
                        ])

y_false2 = torch.tensor([[5.,0.,5.,0.,0.,5.,0.,0.,0.,0.,0.,5.,0.], #légères erreurs hiérarchiques
                         [5.,0.,5.,0.,0.,0.,5.,0.,0.,0.,0.,5.,0.],  
                         [5.,0.,0.,5.,0.,0.,0.,5.,0.,0.,0.,5.,0.]
                        ])

y_false3 = torch.tensor([[5.,0.,5.,0.,0.,0.,5.,0.,0.,0.,5.,0.,0.], #erreurs de prédiction, mais respectant la hiérarchie
                         [5.,0.,5.,0.,0.,5.,0.,0.,0.,5.,0.,0.,0.],  
                         [5.,0.,0.,5.,0.,0.,0.,5.,0.,0.,0.,5.,0.]
                        ])

loss_fn = HierarchicalJsd(hierarchy_csv,hier_weight = 0.1).forward

print("loss pour une prédiction avec de grosses erreurs hiérachiques")
print(loss_fn(y_false1,y_true))

print("loss pour une prédiction avec de légères erreurs hiérachiques")
print(loss_fn(y_false2,y_true))

print("loss pour une prédiction avec des erreurs de prédiction mais pas d'erreurs hiérarchiques")
print(loss_fn(y_false3,y_true))

print("loss pour un résultat exact")
print(loss_fn(y_correct,y_true))
