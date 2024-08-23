from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from Tapioca.segment_hardcode import image_segmenter


class ImageEditor:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Image Editor")
        self.app.geometry("500x600")

        ctk.set_appearance_mode("dark")  # Set the theme

        self.label = ctk.CTkLabel(self.app, text="What modification do you want to do?")
        self.label.pack(pady=10)

        self.modification = ctk.StringVar(value="mSAM")
        self.dropdown = ctk.CTkOptionMenu(self.app, values=["SAM", "mSAM"],
                                          variable=self.modification)
        self.dropdown.pack(pady=10)

        self.weights_button = ctk.CTkButton(self.app, text="Open Weights", command=self.open_weights)
        self.weights_button.pack(pady=10)

        self.results_button = ctk.CTkButton(self.app, text="Results Folder", command=self.open_results_folder)
        self.results_button.pack(pady=10)

        self.open_button = ctk.CTkButton(self.app, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        self.modify_button = ctk.CTkButton(self.app, text="Apply Modification", command=self.modify_image)
        self.modify_button.pack(pady=10, expand=False)

        self.image_frame = ctk.CTkFrame(self.app)
        self.image_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(expand=False, padx=10, pady=10)

        self.app.bind("<Configure>", self.on_resize)
        self.current_image = None

    def open_weights(self):
        weights_path = filedialog.askopenfilename(title="Select Weights File")
        if weights_path:
            print(f"Selected weights file: {weights_path}")
            self.weights = Path(weights_path)

    def open_results_folder(self):
        results_folder = filedialog.askdirectory(title="Select Results Folder")
        if results_folder:
            print(f"Selected results folder: {results_folder}")
            self.results_folder = Path(results_folder)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File")
        if file_path:
            self.current_image = Image.open(file_path)
            self.aspect_ratio = self.current_image.width / self.current_image.height
            self.display_image()
            self.image_path = Path(file_path)

    def display_image(self):
        if self.current_image:
            width = self.image_frame.winfo_width()
            height = self.image_frame.winfo_height()
            img_width, img_height = self.resize_image((self.image_label.winfo_width(),self.image_label.winfo_height()), width, height)
            self.tk_image = ctk.CTkImage(self.current_image, size=(img_width, img_height))
            self.image_label.configure(image=self.tk_image)

    def resize_image(self, image, width, height):
        # Resize the image to fit the frame while maintaining aspect ratio
        print(str(image))
        if width / height > self.aspect_ratio:
            new_height = height -170
            new_width = int(self.aspect_ratio * new_height)

        else:
            new_width = width -170
            new_height = int(new_width / self.aspect_ratio)

        # print(f"new_width: {new_width}, new_height: {new_height}")
        # print(f"width: {width}, height: {height}")
        # print(f"width-new_width: {width-new_width}, height-new_height: {width-new_height}")
        # if new_height > self.app.winfo_height() * .4:
        #     new_height = 200
        #     new_width = int(new_height / self.aspect_ratio)
        # if (image[0] + 50) > width:
        #     new_width = width * .8
        #     new_height = int(new_width / self.aspect_ratio)

        return new_width, new_height

    def on_resize(self, event):
        # Ignore resize events for widgets other than the main window
        if event.widget == self.app:
            # Wait a bit to ensure the frame has been resized
            self.app.after(100, self.display_image)

    def modify_image(self):
        if not self.current_image:
            messagebox.showerror("Error", "Please open an image first.")
            return

        modification = self.modification.get()
        print(f"Applying {modification} modification")
        self.SAM_ob = image_segmenter(self.weights, self.results_folder, modification)
        fin_image = self.SAM_ob.gen_seg(self.image_path)
        # messagebox.showinfo("Modification Applied", f"Applied {modification} to the image")
        self.current_image = fin_image

    def run(self):
        self.app.mainloop()


if __name__ == "__main__":
    editor = ImageEditor()
    editor.run()