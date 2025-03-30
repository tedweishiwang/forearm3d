import torch
import torch.hub
import matplotlib.pyplot as plt

# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)

# Inference
model.eval()
img_rnd = torch.randn(1, 3, 256, 256)  # [B, C, H, W]
preds = model(img_rnd).argmax(1)  # [B, H, W]

# Convert the random image to a format for plotting
# Remove the batch dimension and rearrange to [H, W, C]
img_np = img_rnd.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

# Normalize the image to the range [0, 1]
img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())

# Plot the random image
plt.imshow(img_norm)
plt.title("Random Image")
plt.axis("off")
plt.show()
