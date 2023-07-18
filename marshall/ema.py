import os
import copy

import torch
import torch.nn as nn


class EMA:
    """
    Modified version of class fairseq.models.ema.EMAModule (Exponential Moving Average).
    """

    def __init__(self, model: nn.Module, config, device=None, skip_keys=None):
        """
        Initialize Exponential Moving Average
        """
        self.model = self.deepcopy_model(model)
        self.model.requires_grad_(False)
        self.model.to(device)
        self.config = config
        self.device = device
        self.skip_keys = skip_keys or set()
        self.decay = self.config.model.ema_decay
        self.num_updates = 0

    @staticmethod
    def deepcopy_model(model):
        try:
            model = copy.deepcopy(model)
            return model
        except RuntimeError:
            tmp_path = 'tmp_model_for_ema_deepcopy.pt'
            torch.save(model, tmp_path)
            model = torch.load(tmp_path)
            os.remove(tmp_path)
            return model

    def step(self, new_model: nn.Module):
        """
        One EMA step

        :param new_model: Online model to fetch new weights from
        """
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False)
        self.num_updates += 1

    def restore(self, model: nn.Module):
        """
        Reassign weights from another model

        :param model: model to load weights from
        :return: model with new weights
        """
        d = self.model.state_dict()
        model.load_state_dict(d, strict=False)
        return model

    def state_dict(self):
        return self.model.state_dict()

    @staticmethod
    def get_annealed_rate(start, end, curr_step, total_steps):
        """
        Calculate EMA annealing rate
        """
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining
