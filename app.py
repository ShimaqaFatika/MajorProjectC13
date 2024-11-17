import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import cm as c
import matplotlib.pyplot as plt
from keras.models import model_from_json

class ImagePopup:
    def __init__(self, master):
        self.master = master
        self.master.title("Choose Image")
        self.filename = ""
        self.master.geometry("600x600")  # Set the size of the window
        
        # Browse Button
        self.browse_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=20)
        
        # Submit Button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack(pady=10)
    
    def browse_file(self):
        self.filename = filedialog.askopenfilename()
    
    def submit(self):
        if self.filename:
            self.master.destroy()
            self.show_result()
        else:
            messagebox.showwarning("Warning", "No file selected.")

    def show_result(self):
        count, img, hmap = self.predict(self.filename)

        fig = plt.figure(figsize=(10, 5))
        fig.suptitle("Crowd Counting Result")
        axs = fig.subplots(1, 2)

        axs[0].imshow(img.squeeze())
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(hmap.squeeze(), cmap=c.jet)
        axs[1].set_title(f'Density Map\nEstimated Count: {count}')
        axs[1].axis('off')

        plt.show()
    
    def predict(self, path):
        model = load_model()
        image = create_img(path)
        ans = model.predict(image)
        count = int(np.sum(ans))
        return count, image, ans

def load_model():
    # Function to load and return neural network model 
    json_file = open('/Users/apple/Desktop/CSRnet-master/models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("/Users/apple/Desktop/CSRnet-master/weights/model_A_weights.h5")
    return loaded_model

def create_img(path):
    #Function to load, normalize and return image 
    im = Image.open(path).convert('RGB')
    im = np.array(im)
    im = im / 255.0
    im[:,:,0] = (im[:,:,0] - 0.485) / 0.229
    im[:,:,1] = (im[:,:,1] - 0.456) / 0.224
    im[:,:,2] = (im[:,:,2] - 0.406) / 0.225
    im = np.expand_dims(im, axis=0)
    return im

def main():
    root = tk.Tk()
    app = ImagePopup(root)
    root.mainloop()

if __name__ == "__main__":
    main()
