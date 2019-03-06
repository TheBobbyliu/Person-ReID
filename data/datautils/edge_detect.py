import cv2 
import numpy as np
import torch
def edge_detect(image):
    image = np.array(image)
    edge = cv2.Canny(image,200,300)
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(edge, kernel, iterations = 1)//255
    dilation = cv2.resize(dilation, (384, 128))
    dilation = np.expand_dims(dilation, 0)
    dilation = np.repeat(dilation, 3, axis=0)
    edge = torch.Tensor(dilation)
    return edge
