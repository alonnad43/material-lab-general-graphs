import cv2
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("0cm.png")  # Replace with actual file path
if img is None:
    raise FileNotFoundError("Could not load image")

text = pytesseract.image_to_string(img)
print("OCR Output:")
print(text)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Heatmap Preview")
plt.axis("off")
plt.show()
