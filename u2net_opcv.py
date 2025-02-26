import cv2 as cv 
import numpy as np
import onnxruntime as ort
import torch


def load_onnx_model(model_path, device='cpu'):
    """Loads an ONNX model."""
    if device == 'cuda':
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, providers=providers)
    return ort_session

def preprocess_image(image, input_size=(320, 320)):
    """Preprocesses the image for U2-Net."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.numpy().astype(np.float32)
def preprocess_image_numpy(image_rgb, input_size=(320, 320)):
    """Preprocesses the image for U2-Net using NumPy."""

    # 1. Convert to float32 and normalize to 0-1
    image = image_rgb.astype(np.float32) / 255.0

    # 2. Resize the image
    image = cv.resize(image, input_size, interpolation=cv.INTER_LINEAR)

    # 3. Transpose to (channels, height, width)
    image = np.transpose(image, (2, 0, 1))

    # 4. Normalize with mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image = (image - mean) / std

    # 5. Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)

def generate_trimap(mask, trimap_kernel_size=15):
    """Generates a trimap from a mask."""
    mask_8bit = (mask * 255).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (trimap_kernel_size, trimap_kernel_size))
    dilated = cv.dilate(mask_8bit, kernel, iterations=5)
    eroded = cv.erode(mask_8bit, kernel, iterations=5)
    trimap = np.zeros_like(mask_8bit)
    trimap[dilated == 255] = 128
    trimap[eroded == 255] = 255
    trimap[mask_8bit == 0] = 0
    trimap[mask]=255
    final_trimap=cv.bitwise_or(trimap,mask) 
    cv.imshow('final trimap',final_trimap)
    return trimap

def generate_trimap_binary(mask_path,eroision_iter=1,dilate_iter=1):
     #takes binary image as input
    mask =  mask_path
    mask = cv.imread(mask,0)
    mask[mask==1] = 255
    d_kernel = np.ones((5,5))
    erode  = cv.erode(mask,d_kernel,iterations=eroision_iter)
    dilate = cv.dilate(mask,d_kernel,iterations=dilate_iter)
    unknown1 = cv.bitwise_xor(erode,mask)
    cv.imshow('unknow1',unknown1) 
    unknown2 = cv.bitwise_xor(dilate,mask)
    cv.imshow('unknown2',unknown2) 
    unknowns = cv.add(unknown1,unknown2)
    cv.imshow('unknowns',unknowns)
    unknowns[unknowns==255]=127
    cv.imshow('unknowns',unknowns)
    trimap = cv.add(mask,unknowns)
    # cv2.imwrite("mask.png",mask)
    # cv2.imwrite("dilate.png",dilate)
    # cv2.imwrite("tri.png",trimap)
    labels = trimap.copy()
    labels[trimap==127]=1 #unknown
    labels[trimap==255]=2 #foreground
    #cv2.imwrite(mask_path,labels)
    return labels

def postprocess_mask(mask, original_size):
    """Postprocesses the mask to the original image size."""
    mask = cv.resize(mask, (original_size[1], original_size[0]), interpolation=cv.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    return mask

def remove_background(image, model_path, output_path, device='cpu'):
    """Removes the background from an image using U2-Net ONNX."""
    image = cv.imread(image_path)
    original_size = image.shape[:2]
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    ort_session = load_onnx_model(model_path, device)

    # input_tensor = preprocess_image(torch.from_numpy(image_rgb).permute(2, 0, 1)) # changed to permute
    input_tensor=preprocess_image_numpy(image_rgb)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    mask = ort_outs[0][:, 0, :, :] # output of the onnx model.
    print('original shape',image.shape) 
    mask = postprocess_mask(mask[0], original_size) 
    print('mask image ',mask.shape)

    cv.imwrite('mask.png',mask)
    mask = (mask > 0.5).astype(np.uint8) * 255
    cv.imshow('mask ',mask)
    binary_trimap=generate_trimap_binary('mask.png')
    cv.imshow('binary_trimap',binary_trimap) 

    # alphavalue=0.3
    # for x in range(mask_image.shape[0]): 
    #     for y in range(mask_image.shape[1]):
    #         alpha[x][y]=alphavalue*image[x][y]+(1-alphavalue)*image[x][y]
    # cv.imshow('alpha image',alpha)

    # trimap=generate_trimap(mask) 
    # cv.imshow(
    #     'trimap', trimap
    # )
    # # Create a mask with 3 channels
    # mask_3channel = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # # Apply the mask to the image
    # masked_image = cv.bitwise_and(image, mask_3channel)
    # cv.imshow('masked_image',masked_image)
    # # Create a white background
    # white_background = np.full_like(image, 255)

    # # Combine the masked image with the white background
    # background_removed_image = np.where(mask_3channel > 0, masked_image, white_background)
    # cv.imwrite(output_path, background_removed_image)
    cv.waitKey(0)
    cv.destroyallwindows()

if __name__ == "__main__":
    image_path = "input.jpg"  # Replace with your image path
    model_path = "u2net.onnx"  # Replace with your ONNX model path
    output_path = "output.png"  # Replace with your desired output path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use CUDA if available

    remove_background(image_path, model_path, output_path, device)
    print(f"Background removed and saved to {output_path}")