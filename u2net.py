import torch 
import cv2
import numpy as np
import onnxruntime as ort;
import os;
from PIL import Image
from skimage import io

def normPRED(d):
    d=torch.from_numpy(d)
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn
def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    print(d_dir+imidx)
    imo.save(d_dir+imidx+'.png')

def preprocess_input(image_path, target_size=(320, 320)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change from HWC to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():

    prediction_dir = os.path.join(os.getcwd())
    sess = ort.InferenceSession('u2net.quant.onnx') 
    input_name = sess.get_inputs()[0].name
    #model=onnx.load('u2net.quant.onnx')
    input_image = preprocess_input("1.jpg")
    output = sess.run(None, {input_name: input_image.astype(np.float32)})
    #cv2.imshow("outou",img)
    d1,d2,d3,d4,d5,d6,d7=output
    
    pred = d1[:,0,:,:]
    print(pred.shape,type(pred.shape))
    pred = normPRED(pred)    
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    save_output("1.png",pred,prediction_dir)



if __name__=="__main__":
    main()