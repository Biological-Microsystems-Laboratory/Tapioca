import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk, ImageEnhance

from main import image_segmenter


class ImageEditor:
    def __init__(self, master):
        self.master = master
        master.title("Image Editor")

        self.label = tk.Label(master, text="What modification do you want to do?")
        self.label.pack()

        self.modification = tk.StringVar(master)
        self.modification.set("mSAM")  # default value
        self.dropdown = tk.OptionMenu(master, self.modification, "mSAM", "SAM2", "SAM")
        self.dropdown.pack()

        self.open_button = tk.Button(master, text="Open Image", command=self.open_image)
        self.open_button.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.modify_button = tk.Button(master, text="Apply Modification", command=self.modify_image)
        self.modify_button.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        image.thumbnail((400, 400))  # Resize image for display
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.tk_image)

    def modify_image(self):
        if not hasattr(self, 'original_image'):
            messagebox.showerror("Error", "Please open an image first.")
            return

        modification = self.modification.get()
        SAM_obj = image_segmenter(MODEL=modification)

        modified_image = SAM_obj.gen_seg(Path(self.image_path))
        im_pil = Image.fromarray(modified_image)

        self.display_image(im_pil)

root = tk.Tk()
editor = ImageEditor(root)
root.mainloop()