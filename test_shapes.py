import torch
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(26, 512)
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
dummy = torch.randn(1, 26)
torch.onnx.export(model, dummy, "test.onnx", opset_version=14)
print("Export OK")
