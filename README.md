# Workshop -1 Adding Sunglasses to Your Passport Photo Using OpenCV
# Name : VAISHNAVI S
# Reg No : 212225230289
## Program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

faceImage = cv2.imread('img2.jpeg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

faceImage.shape
#resized_faceImage.shape
faceImage.shape

# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(280,140))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[240:380,272:552]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[240:380,272:552]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")

# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[240:380,272:552]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```
## Output
<img width="501" height="596" alt="image" src="https://github.com/user-attachments/assets/d8c80d45-9cd5-449f-bbca-de009d89b4b7" />
<img width="733" height="580" alt="image" src="https://github.com/user-attachments/assets/ad8cc539-0e60-4268-91ff-da3ad35abc86" />
<img width="907" height="345" alt="image" src="https://github.com/user-attachments/assets/359a9d24-4fc6-4eab-8424-159b5f606168" />
<img width="487" height="575" alt="image" src="https://github.com/user-attachments/assets/c412fe9f-2709-472c-bac5-3e93c52460b8" />
<img width="903" height="272" alt="image" src="https://github.com/user-attachments/assets/e52e5b61-98ba-4f09-9dd8-7ce2cfaf4c6f" />
<img width="905" height="497" alt="image" src="https://github.com/user-attachments/assets/6416863b-109f-4340-86e3-0e6dfc14be93" />

## Result
Thus, adding sunglass to the passport photo using opencv is performed successfully.
