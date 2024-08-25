import os
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable

from metrics.subjects.linearity.src.model import Model as Linearity

DEVICE = os.environ.get("DEVICE", "cuda")
print("DEVICE=" + DEVICE)

def adv(ref, eps):
    image = transforms.ToTensor()(ref)
    image = image.unsqueeze_(0)
    image = image.to(DEVICE)
    image = Variable(image, requires_grad=True)

    model = Linearity(DEVICE)
    score = model(image)
    print("linearity score:", score.item())
    loss = 1 - score / 100
    loss.backward() 
    g = image.grad
    g = torch.sign(g)
    image.data -= eps * g
    image.data.clamp_(0., 1.)
    image.grad.zero_()

    res_image = (image).data.clamp_(min=0, max=1)
    res_img = (res_image.squeeze().data.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
    return res_img

images = sorted(os.listdir('test_ims'))
im = cv2.imread('test_ims/'+images[0])
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
eps = 10/255
im = adv(im, eps)
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
os.makedirs('res_fgsm', exist_ok=True)
cv2.imwrite('res_fgsm/'+images[0], im)
