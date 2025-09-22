import numpy as np
import cv2
import logging
import os

import time



class WeldingCheck:
    """
    Checks if welding is detected in the image based on median L value
    """
    def __init__(self, threshold_torch=70):
        self.threshold_torch = threshold_torch

    def median_L_from_RGB(self, image):
        """
        Converts an RGB8 image to HLS and returns the median L value 
        Calcultes directly with numpy for speed
        0-255 scale
        """
        R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
        Cmax = np.maximum(np.maximum(R, G), B)
        Cmin = np.minimum(np.minimum(R, G), B)
        median_L = np.median((Cmax + Cmin) / 2)
        return median_L
    
    def is_welding(self, image):    
        """
        Public method to check if welding is detected
        Can be called by other classes
        """
        median_L = self.median_L_from_RGB(image)
        return median_L > self.threshold_torch
    def run(self, image, analyzer):
        median_L = self.median_L_from_RGB(image)
        is_welding = median_L > self.threshold_torch
        
        #logging.info(f"{analyzer.current_file}: Median L: {median_L:.2f}, Welding: {is_welding}")
        
        return {"welding": is_welding, "median_L_welding": round(median_L, 2)} 


class SharpnessCheck:
    """
    Checks sharpness using the variance of the Laplacian
    If the variance is below a certain threshold, the image is considered blurry and not usable
    Its only checked if welding is detected in the image
    
    
    """
    def __init__(self, threshold_sharp=12.0):
        """
        Laplacian Variance below 12 is considered blurry could be adjusted 
        """
        self.threshold_sharp = threshold_sharp
        self.welding_check = WeldingCheck()
    def run(self, image, analyzer):
        # if not analyzer.check_welding(image):
        #     return {"sharpness": None, "blurry": False}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        is_blurry = lap_var < self.threshold_sharp
        is_welding = self.welding_check.is_welding(image)


        if is_blurry and is_welding:
            logging.warning(f"{analyzer.current_file}: Blurry! (Laplacian Variance: {lap_var:.2f})")
            analyzer.flags += 1

        return {"sharpness": round(lap_var, 2), "blurry": is_blurry}


class DarknessCheck:
    """
    Coverts the RGB8 image into HLS and checks the median L value
    If the median L value is below 25, the image is considered too dark
    """
    def __init__(self, threshold_dark=25.0):
        self.threshold_dark = threshold_dark
        
    def median_L_from_RGB(self, image):
        """
        Converts an RGB8 image to HLS and returns the median L value 
        Calcultes directly with numpy for speed
        0-255 scale
        
        """

        R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
        Cmax = np.maximum(np.maximum(R, G), B)
        Cmin = np.minimum(np.minimum(R, G), B)
        median_L = np.median((Cmax + Cmin) / 2)
        return median_L
    
    def run(self, image, analyzer):
        median_L = self.median_L_from_RGB(image)
        is_dark = median_L < self.threshold_dark
        
        if is_dark:
            logging.warning(f"{analyzer.current_file}: Too dark! (Median L: {median_L:.2f})")
            analyzer.flags += 1
            
        return {"median_L": round(median_L, 2), "too_dark": is_dark}


class TorchPositionCheck:
    """
    Checks if the torch is in the correct position by finding the largest contour in the image
    and checking if its centroid is in the middle within tolerance.
    """

    def __init__(self, tolerance=0.15):
        self.tolerance = tolerance
        self.welding_check = WeldingCheck()
        self.bright_mask_threshold = 200  

    def rgb_to_hls(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
    def run(self, image, analyzer):
        # Check if welding is detected first
        if not self.welding_check.is_welding(image):
            return {"torch_in_middle": False, "torch_position": None, "reason": "no_welding"}

        height, width = image.shape[:2]
        cx_target, cy_target = width / 2, height / 2
        tol_x, tol_y = width * self.tolerance, height * self.tolerance

        image_hls = self.rgb_to_hls(image)
        L = image_hls[:, :, 1]
        bright_mask = (L > self.bright_mask_threshold).astype(np.uint8)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"torch_in_middle": False, "torch_position": None, "reason": "no_contours"}

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return {"torch_in_middle": False, "torch_position": None, "reason": "invalid_moments"}

        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        is_in_middle = (abs(cx - cx_target) < tol_x) and (abs(cy - cy_target) < tol_y)
        
        if not is_in_middle:
            logging.warning(f"{analyzer.current_file}: Torch not in middle! Position: ({cx}, {cy})")
            analyzer.flags += 1
            
        return {"torch_in_middle": is_in_middle, "torch_position": f"({cx}, {cy})"}
    





    
class ImageAnalyzer:
    """
    The whole purpose of this class is to call the different checks and store the results in a csv file
    This class doesnt do any image processing itself 
    """
    def __init__(self, folder_path, flags=0, output_csv="analysis_results.csv"):
        self.folder_path = folder_path
        self.flags = flags
        self.output_csv = output_csv
        self.current_file = None
        
        # Put your desired checks here
        self.checks = [
            WeldingCheck(),
            SharpnessCheck(),
            DarknessCheck(),
            TorchPositionCheck(),
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def read_image(self, path):
        """
        Read an image from the given path
        """
        image = cv2.imread(path)
        if image is None:
            logging.warning(f"Picture couldn't be loaded: {path}")
        return image

    def analyze_folder(self):
        """
        main function to analyze all images in the folder
        1. Reads each image
        2. Applies each check in self.checks
     
        """
        results = []

        for filename in os.listdir(self.folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(self.folder_path, filename)
            image = self.read_image(path)
            if image is None:
                continue

            self.current_file = filename
            file_result = {"filename": filename}

            for check in self.checks:
                file_result.update(check.run(image, self))

            results.append(file_result)



        if self.flags == 0:
            logging.info("Alle Bilder sind in Ordnung.")
        elif self.flags <= 6:
            logging.info(f"Es wurden {self.flags} Probleme gefunden, höchstwahrscheinlich sind die Bilder in Ordnung.")
        else:
            logging.warning(f"Es wurden {self.flags} Probleme gefunden.")



if __name__ == "__main__":
    folder = r"C:\Users\justu\Desktop\2025-08-19_23-28-24_9431f598\2025-08-19_23-28-24_9431f598"
    start_time = time.time()
    analyzer = ImageAnalyzer(folder_path=folder)
    analyzer.analyze_folder()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Analyse abgeschlossen in {elapsed_time:.2f} Sekunden.")
