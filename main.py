import numpy as np
import cv2
import os
import csv


"""
If you want to check a whole folder for a specific problem, just add the corresponding check class to self.checks in ImageAnalyzer
line 88

"""


class SharpnessCheck:
    """
    Checks sharpness using the variance of the Laplacian
    If the variance is below a certain threshold, the image is considered blurry and not usable
    
    
    """
    def run(self, image, analyzer):
        # if not analyzer.check_welding(image):
        #     return {"sharpness": None, "blurry": False}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        print(lap_var)
        is_blurry = lap_var < analyzer.threshold_sharp

        if is_blurry:
            print(f"{analyzer.current_file}: Unscharf! (Laplacian Variance: {lap_var:.2f})")
            analyzer.flags += 1

        return {"sharpness": round(lap_var, 2), "blurry": is_blurry}



class DarknessCheck:
    """
    Coverts the RGB8 image into HLS and checks the median L value
    If the median L value is below 25, the image is considered too dark and not usable
    
    """
    def run(self, image, analyzer):
        median_L = analyzer.median_L_from_RGB(image)
        is_dark = median_L < 25

        if is_dark:
            print(f"{analyzer.current_file}: DARK! (Median L: {median_L:.2f})")
            analyzer.flags += 1

        return {"median_L": round(median_L, 2), "dark": is_dark}



class TorchPositionCheck:
    """
    Checks the position of the torch in the image

    """
    def run(self, image, analyzer):
        if not analyzer.check_welding(image):
            return {"torch_middle": False}

        in_middle = analyzer.torch_in_middle(image)
        if not in_middle:
            print(f"{analyzer.current_file}: Fackel nicht in der Mitte.")
            analyzer.flags += 1

        return {"torch_middle": in_middle}



class ImageAnalyzer:
    """
    Iterates through every image in a given folder and applies the checks
    defined in self.checks. Results are saved to a CSV file.
    """
    def __init__(self, folder_path, threshold_torch=70, threshold_sharp=10, flags=0, output_csv="analysis_results.csv"):
        self.folder_path = folder_path
        self.threshold_torch = threshold_torch
        self.threshold_sharp = threshold_sharp
        self.flags = flags
        self.output_csv = output_csv
        self.current_file = None

        # Put your desired checks here
        self.checks = [
            SharpnessCheck(),
            #TorchPositionCheck(),
            #DarknessCheck(),
            
        ]

    def read_image(self, path):
        """
        Read an image from the given path
        """
        image = cv2.imread(path)
        if image is None:
            print(f"Bild konnte nicht geladen werden: {path}")
        return image

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

    def check_welding(self, image):
        """
        Welding is detected if the median L value is above a certain threshold
        """
        median_L = self.median_L_from_RGB(image)
        return median_L > self.threshold_torch

    def rgb_to_hls(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    def torch_in_middle(self, image, tolerance=0.15):
        """
        Checks if the torch is in the middle of the image within a certain tolerance
        This is done by using cv2.findContours on a bright mask created from the L channel in HLS
        This funcion calculates the centroid of the largest bright contour and checks if it is within the tolerance of the image center
        
        
        """
        if not self.check_welding(image):
            return False

        height, width = image.shape[:2]
        cx_target, cy_target = width / 2, height / 2
        tol_x, tol_y = width * tolerance, height * tolerance

        image_hls = self.rgb_to_hls(image)
        L = image_hls[:, :, 1]
        bright_mask = (L > 200).astype(np.uint8)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return False

        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return (abs(cx - cx_target) < tol_x) and (abs(cy - cy_target) < tol_y)

    
    def analyze_folder(self):
        """
        main function to analyze all images in the folder
        1. Reads each image
        2. Applies each check in self.checks
        3. Saves results to a CSV file
        4. Prints summary of findings

        
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
            is_welding = self.check_welding(image)
            file_result = {"filename": filename, "welding": is_welding}

            for check in self.checks:
                file_result.update(check.run(image, self))

            results.append(file_result)

        if results:
            with open(self.output_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            print(f"Analyse abgeschlossen. Ergebnisse in {self.output_csv} gespeichert.")

        if self.flags == 0:
            print("Alle Bilder sind in Ordnung.")
        elif self.flags <= 6:
            print(f"Es wurden {self.flags} Probleme gefunden, höchstwahrscheinlich sind die Bilder in Ordnung.")
        else:
            print(f"Es wurden {self.flags} Probleme gefunden.")


if __name__ == "__main__":
    folder = r"C:\Users\justu\Desktop\Image_Analyzer\Schaerfestufen"
    analyzer = ImageAnalyzer(folder_path=folder, threshold_torch=60, threshold_sharp=12.0)

    # Fall 1: Alle Checks
    analyzer.analyze_folder()

    # Fall 2: Nur einzelne Checks benutzen
    # test_img = analyzer.read_image(r"C:\Users\justu\Desktop\Neuer Ordner\Plasmafackel\Frame_0016.jpeg")
    # analyzer.current_file = "Frame_0016.jpeg"  # für konsistente Ausgaben

    # sharpness_result = SharpnessCheck().run(test_img, analyzer)
    # print("Nur Schärfe:", sharpness_result)

    #darkness_result = DarknessCheck().run(test_img, analyzer)
    #print("Nur Dunkelheit:", darkness_result)

    #torch_result = TorchPositionCheck().run(test_img, analyzer)
    #print("Nur Torch-Mitte:", torch_result)
