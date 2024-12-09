import copy
import importlib

import torch
import torch.nn as nn
from collections.abc import Mapping
import cv2
import numpy as np
from ngp.gridencoder import GridEncoder
from ngp.ffmlp import FMLP

class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)
            break

    def build(self, mtype, kwargs):
        if mtype == "torch_ngp.GridEncoder":
            from ngp.gridencoder import GridEncoder
            return GridEncoder(**kwargs)
        elif mtype == "torch_ngp.FMLP":
            from ngp.ffmlp import FMLP
            return FMLP(**kwargs)
        else:
            module_name, cls_name = mtype.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        input = copy.copy(input)
        if input.device != self.device:
            input = input.to(self.device)
        
        for submodule in self.children():
            if isinstance(submodule, GridEncoder):
                input = submodule(input)  # Encode input with GridEncoder
            elif isinstance(submodule, FMLP):
                input = submodule(input)  # Process encoded input with FMLP
            else:
                input = submodule(input)  # Process with other submodules

        return input  # Return the processed result


    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
