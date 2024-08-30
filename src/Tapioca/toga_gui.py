import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from pathlib import Path

from Tapioca.segment_hardcode import image_segmenter

class ImageEditor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Editor")
        self.root.geometry("500x600")

        self.label = tk.Label(self.root, text="What modification do you want to do?")
        self.label.pack(pady=10)

        self.modification = tk.StringVar(value="mSAM")
        self.dropdown = ttk.Combobox(self.root, textvariable=self.modification, values=["SAM", "mSAM"])
        self.dropdown.pack(pady=10)

        self.weights_button = tk.Button(self.root, text="Open Weights", command=self.open_weights)
        self.weights_button.pack(pady=10)

        self.results_button = tk.Button(self.root, text="Results Folder", command=self.open_results_folder)
        self.results_button.pack(pady=10)

        self.open_button = tk.Button(self.root, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        self.modify_button = tk.Button(self.root, text="Apply Modification", command=self.modify_image)
        self.modify_button.pack(pady=10)

        self.scale_input = tk.Entry(self.root,text="Scale (px/um)")
        self.scale_input.pack(pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.root.bind("<Configure>", self.on_resize)
        self.current_image = None

    def open_weights(self):
        weights_path = filedialog.askopenfilename(title="Select Weights File")
        if weights_path:
            print(f"Selected weights file: {weights_path}")
            self.weights = Path(weights_path)

    def set_scale(self):
        self.image_scale = self.scale_input.get()
        print(f"the scale is {self.image_scale}")

    def open_results_folder(self):
        results_folder = filedialog.askdirectory(title="Select Results Folder")
        if results_folder:
            print(f"Selected results folder: {results_folder}")
            self.results_folder = Path(results_folder)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File")
        if file_path:
            if file_path.endswith((".tiff", ".tif")):
                messagebox.showinfo("Modification Applied", f"Image path: {file_path} but its not going to show")
                self.image_path = Path(file_path)
            self.current_image = Image.open(file_path)
            self.aspect_ratio = self.current_image.width / self.current_image.height
            self.display_image()
            self.image_path = Path(file_path)

    def display_image(self):
        if self.current_image is not None:
            width = self.image_frame.winfo_width()
            height = self.image_frame.winfo_height()
            img_width, img_height = self.resize_image((self.image_label.winfo_width(), self.image_label.winfo_height()), width, height)
            resized_image = self.current_image.copy()
            resized_image.thumbnail((img_width, img_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_image)
            self.image_label.config(image=self.tk_image)
            self.image_label.image = self.tk_image  # Keep a reference

    def resize_image(self, image, width, height):
        max_width = width - 20  # Subtract padding
        max_height = height - 20  # Subtract padding
        scale = min(max_width / self.current_image.width, max_height / self.current_image.height)
        new_width = int(self.current_image.width * scale)
        new_height = int(self.current_image.height * scale)
        return new_width, new_height

    def on_resize(self, event):
        if event.widget == self.root:
            self.root.after(100, self.display_image)

    def modify_image(self):
        set_scale()
        if not hasattr(self, 'image_path'):
            messagebox.showerror("Error", "Please open an image first.")
            return

        modification = self.modification.get()
        print(f"Applying {modification} modification")
        try:
            self.SAM_ob = image_segmenter(self.weights, self.results_folder, modification, SCALE=int(self.image_scale))
            fin_image = self.SAM_ob.gen_seg(self.image_path)
            self.aspect_ratio = fin_image.shape[1] / fin_image.shape[0]
            self.current_image = Image.fromarray(fin_image)
            self.display_image()
            messagebox.showinfo("Success", f"Applied {modification} to the image")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply modification: {str(e)}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    editor = ImageEditor()
    editor.run()