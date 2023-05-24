import shutil

import torch
import torch._dynamo
import torch._inductor


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x, y):
        return self.fc(torch.sin(x) + torch.cos(y))


x = torch.randn((32, 64), device="cuda")
y = torch.randn((32, 64), device="cuda")

for dynamic in [True, False]:
    torch._dynamo.config.dynamic_shapes = dynamic
    torch._dynamo.reset()

    with torch.no_grad():
        module, _ = torch._dynamo.export(Net().cuda(), x, y)
        lib_path = torch._inductor.aot_compile(module, [x, y])

    shutil.copy(lib_path, f"libaot_inductor_output{'_dynamic' if dynamic else ''}.so")
