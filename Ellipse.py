import cv2
import numpy as np
import os



image = cv2.imread(r"C:\Users\justu\Desktop\Bilder\Neuer Ordner\2025-08-19_23-23-48_35b65e92\Frame_0000.jpeg")
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_blur = cv2.GaussianBlur(image_gray,(39,39),0)
sobel_y = cv2.Sobel(image_blur,cv2.CV_64F,0,1,ksize=5)

#Now we try to find the biggest contour in the image
_,thresh = cv2.threshold(sobel_y,50,255,cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Nur Ellipse fitting und zeichnen
    if len(largest_contour) >= 100:  
        ellipse = cv2.fitEllipse(largest_contour)
        angle = ellipse[2]
        if angle > 95 or angle < 85:
            print("Winkel außerhalb des Bereichs!")
        print(f"Ellipse Winkel: {angle:.1f}°")
        cv2.ellipse(image, ellipse, (0, 0, 255), 2)
cv2.imshow("Image",image)


cv2.imwrite("output.jpeg",image)
cv2.imwrite("Kantenerkennung_Maske.jpeg",thresh)
cv2.imshow("Gray",sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()