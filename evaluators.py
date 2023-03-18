import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from scipy.optimize import linear_sum_assignment

bceloss = nn.BCELoss(reduction='none')
img2cross_entropy = nn.CrossEntropyLoss()
thre_list = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
img2mse = lambda x, y: torch.mean((x - y) ** 2)
nllloss = nn.NLLLoss(ignore_index=-1, reduction='none')
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
cross_entropy = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")