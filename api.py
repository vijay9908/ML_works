import io
import json
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask,jsonify,request
import socket

app = Flask(__name__)

imagenet_class_index = json.load(open('image_class_index.json'))

model = torch.load('webtry.pth')
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


#image transform functions

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(),transforms.ToTensor(),transforms.Normalize(
    [0.485,0.456,0.406],[0.229,0.224,0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


#get prediction

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor.to(device))
    _,y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id , class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id':class_id,'class_name':class_name})


if __name__ == "__main__":
    app.run()
    
