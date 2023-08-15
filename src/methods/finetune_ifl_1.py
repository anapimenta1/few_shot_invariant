import argparse
import torch
from typing import Dict, Tuple
import torch.nn.functional as F

from .utils import get_one_hot, extract_features
from .method import FSmethod

class Finetune_IFL_1(FSmethod):
    def __init__(self,
                 args: argparse.Namespace):
        self.iter = args.iter
        self.lr = args.finetune_lr
        self.finetune_all_layers = args.finetune_all_layers
        self.normalize = args.normalize
        super().__init__(args)

    def forward(self,
                model: torch.nn.Module,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                task_ids: Tuple[int, int] = None):
        device = x_s.device
        if self.finetune_all_layers:
            model.train()
        else:
            model.eval()
        n_tasks = x_s.size(0)
        if n_tasks > 1:
            raise ValueError('Finetune method can only deal with 1 task at a time. \
                             Currently {} tasks.'.format(n_tasks))
        y_s = y_s[0]
        y_q = y_q[0]
        num_classes = y_s.unique().size(0)
        y_s_one_hot = get_one_hot(y_s, num_classes)

        # Initialize classifier
        with torch.no_grad():
            z_s = extract_features(x_s, model)
            if self.normalize:
                z_s = F.normalize(z_s, dim=-1)
            classifier = torch.nn.Linear(z_s.size(-1), num_classes, bias=True).to(device)

        # Define optimizer
        if self.finetune_all_layers:
            params = list(model.parameters()) + list(classifier.parameters())
        else:
            params = classifier.parameters()
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Run adaptation
        with torch.set_grad_enabled(True):
            for i in range(1, self.iter):
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
                if self.normalize:
                    z_s = F.normalize(z_s, dim=-1)
                    z_q = F.normalize(z_q, dim=-1)

                #print(f"iteration {i}: z_s = {z_s}")

                # Compute the mean feature vector for each class
                class_means = []
                for c in range(num_classes):
                    class_mask = (y_s == c)
                    class_mean = z_s[0][class_mask].mean(0)
                    class_means.append(class_mean)

                # Compute pairwise differences between class means
                invariant_loss = 0
                for k in range(num_classes):
                    for j in range(k+1, num_classes):
                        invariant_loss += (class_means[k] - class_means[j]).pow(2).sum()

                # Combine with existing loss
                invariant_weight = 0.1  # Hyperparameter to weight the importance of the invariant loss
                logits_s = classifier(z_s[0])
                cross_entropy_loss = F.cross_entropy(logits_s, y_s)
                total_loss = cross_entropy_loss + invariant_weight * invariant_loss

                #print(f"iteration {i}, loss = {total_loss}")
                #print(f"iteration {i}, cross_entropy_loss = {cross_entropy_loss}")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        probs_q = classifier(z_q[0]).softmax(-1).unsqueeze(0)
        return total_loss.detach(), probs_q.detach().argmax(2)
