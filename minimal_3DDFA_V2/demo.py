from TDDFA.TDDFA import TDDFA
import cv2

# Initialize model
tddfa = TDDFA(
    weights='weights/mb1_120x120.pth',
    bfm_path='weights/bfm_no_neck_v3.pkl', 
    bfm_tri_path='weights/bfm_tri.pkl',
    _3DDM_mean_std_path='weights/param_mean_std_62d_120x120.pkl',
    gpu_mode='gpu',
)

# Load in inputs
img = cv2.imread("data/sample.png")
H, W = img.shape[:2]
boxes = [[200, 200, 600, 700]]

# Run model, get 3DDM parameters
param_lst, roi_boxes = tddfa(img, boxes)

# Reconstruct image space landmarks from 3DDM
outputs = tddfa.recon_vers(param_lst, roi_boxes)

# Draw landmarks
for landmarks in outputs:
    D, N = landmarks.shape
    for i in range(N):
        x = int(landmarks[0, i])
        y = int(landmarks[1, i])
        img = cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

cv2.imwrite('data/output.jpg', img)
