import torch
import torch.nn as nn


class FastFlowLoss(nn.Module):
    def forward(self, outputs, log_jacobians):
        loss = torch.tensor(0.0, device=outputs[0].device)
        for output, log_jacobian in zip(outputs, log_jacobians):
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jacobian
            )

        return loss
