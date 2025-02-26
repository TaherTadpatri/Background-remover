import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# Load the pre-trained P3M-Net model
model = torch.load('P3M-Net_ViTAE-S_trained_on_P3M-10k.pth', map_location=torch.device('cpu'), weights_only=False) # Ensure weights_only=False

model.eval()

# Define the preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def remove_background(image_path, output_path):
    """Removes the background from an image using P3M-Net."""

    # Load the image using OpenCV
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    original_size = (original_image.shape[1], original_image.shape[0]) # (width, height)

    # Convert to RGB if needed
    if original_image.shape[2] == 4:
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGRA2RGB)
    else:
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    img_tensor = transform(torch.from_numpy(image_rgb).permute(2, 0, 1)).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        alpha, fg = model(img_tensor)

    # Postprocess the alpha matte
    alpha = alpha.squeeze().cpu().numpy()
    alpha = cv2.resize(alpha, original_size, interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    # Convert original image to RGBA if needed
    if original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)

    # Apply the alpha matte
    original_image[:, :, 3] = alpha

    # Save the result
    cv2.imwrite(output_path, original_image)

# Example usage:
image_path = 'input.jpg'  # Replace with your image path
output_path = 'output.png' # Replace with your output path

remove_background(image_path, output_path)
print(f"Background removed and saved to {output_path}")