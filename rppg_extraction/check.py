from torchinfo import summary
from model import *
model = DeepPhys()
summary(model)