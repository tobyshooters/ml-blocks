import cv2
import torch
from facemesh import FaceMesh

# Initialize model
net = FaceMesh()
net.initialize('weights.pth', 'cuda')

# Load in input, crop out face, resize to 192 pixels
img = cv2.imread("data/sample.png")
img = img[200:700, 200:600]
img = cv2.resize(img, (192, 192))

# Get output
mesh = net.process(img)

# Draw landmarks
for landmark in mesh:
    x = int(landmark[0])
    y = int(landmark[1])
    img = cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

cv2.imwrite("data/output.jpg", img)
