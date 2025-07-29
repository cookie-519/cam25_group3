import torch

# 加载完整模型
model = torch.load("model/generator.pth", map_location="cpu")
model.eval()

# 保存为仅包含权重的 state_dict 格式
torch.save(model.state_dict(), "model/generator_state.pth")

print("✅ 已成功转换为 state_dict 格式")
