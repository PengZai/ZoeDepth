# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config, change_dataset
from pprint import pprint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

# Trigger reload of MiDaS


# model_K = torch.hub.load(".", "ZoeD_K", source="local", pretrained=True).to(DEVICE)
# model_N = torch.hub.load("i.", "ZoeD_N", source="local", pretrained=True).to(DEVICE)
# model_NK = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True).to(DEVICE)

# kconf = get_config("zoedepth", "infer", config_version="kitti")
# kconf = change_dataset(kconf, new_dataset="kitti")
# print("Config:")
# print(kconf)
# model_K = build_model(kconf).to(DEVICE)

# nconf = get_config("zoedepth", "infer")
# print("Config:")
# print(nconf)
# model_N = build_model(nconf).to(DEVICE)


nkconf = get_config("zoedepth_nk", "infer")
print("Config:")
print(nkconf)
model_NK = build_model(nkconf).to(DEVICE)





with torch.no_grad():

    img = torch.rand(1, 3, 384, 512).to(DEVICE)
    out = model_NK(img)[1]
    # traced_script_module_for_zoeD_K = torch.jit.trace(model_K, img)
    # traced_script_module_for_zoeD_K.save("ZoeD_K_traced.pt")

    # traced_script_module_for_zoeD_N = torch.jit.trace(model_N, img)
    # traced_script_module_for_zoeD_N.save("ZoeD_N_traced.pt")

    traced_script_module_for_zoeD_NK = torch.jit.trace(model_NK, [img])
    traced_script_module_for_zoeD_NK.save("ZoeD_NK_traced.pt")

