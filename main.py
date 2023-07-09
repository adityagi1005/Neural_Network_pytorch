from torchnn import *

with open('model_state.pt', 'rb') as f: 
    clf.load_state_dict(load(f))  

img = Image.open('img_1.jpg') 
img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

print(torch.argmax(clf(img_tensor)))