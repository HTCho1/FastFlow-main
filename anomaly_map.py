import torch
import torch.nn.functional as F


class AnomalyMapGenerator:

    def __call__(self, outputs, input_size):
        anomaly_map_list = []
        for output in outputs:
            log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            a_map = F.interpolate(
                input=-prob,
                size=[input_size, input_size],
                mode='bilinear',
                align_corners=False
            )
            anomaly_map_list.append(a_map)
        anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
        anomaly_map = torch.mean(anomaly_map_list, dim=-1)

        return anomaly_map
