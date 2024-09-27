import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk, Image
from pathlib import Path
import cv2
import tifffile as tiff
import numpy as np

from segment_hardcode import image_segmenter

def open_tiff_image(file_path):
    tiff_image = tiff.imread(file_path)
    if tiff_image.dtype != np.uint8:
        tiff_image = cv2.normalize(tiff_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    if len(tiff_image.shape) == 3:
        img_rgb = cv2.cvtColor(tiff_image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = tiff_image
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.convert("RGB")
    return pil_img

def normalize(image):

    image = image.astype('uint8')
    # image = cv2.cvtColor(image, cv2.COLOR_BRG2RGB)
    return image


class ImageEditor:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Tapioca")
        self.root.geometry("600x700")
        self.root.configure(bg="#2E2E2E")  # Dark background

        # Title Label
        # self.label = tk.Label(self.root, text="Image Editor", font=("Arial", 18, "bold"), bg="#2E2E2E", fg="#ffffff")
        # self.label.pack(pady=20)

        # Dropdown Frame
        self.dropdown_frame = tk.Frame(self.root, bg="#2E2E2E")
        self.dropdown_frame.pack(pady=10, padx=20, fill="x")
        tk.Label(self.dropdown_frame, text="Modification Type:", bg="#2E2E2E", fg="#ffffff", font=("Arial", 12)).pack(side="left")
        self.modification = tk.StringVar(value="mSAM")
        self.dropdown = ttk.Combobox(self.dropdown_frame, textvariable=self.modification, values=["SAM", "mSAM"])
        self.dropdown.pack(side="left", padx=10)

        # Buttons Frame
        self.buttons_frame = tk.Frame(self.root, bg="#2E2E2E")
        self.buttons_frame.pack(pady=20, padx=20, fill="x")

        button_style = {
            "bg": "#444444",  # Dark gray for button background
            "fg": "#ffffff",  # White text
            "activebackground": "#666666",  # Lighter gray when active
            "activeforeground": "#ffffff",  # White text when active
            "width": 15
        }

        self.weights_button = tk.Button(self.buttons_frame, text="Open Weights", command=self.open_weights, **button_style)
        self.weights_button.pack(side="left", padx=10)

        self.results_button = tk.Button(self.buttons_frame, text="Results Folder", command=self.open_results_folder, **button_style)
        self.results_button.pack(side="left", padx=10)

        self.open_button = tk.Button(self.buttons_frame, text="Open Image", command=self.open_image, **button_style)
        self.open_button.pack(side="left", padx=10)

        self.modify_button = tk.Button(self.buttons_frame, text="Apply Modification", command=self.modify_image, **button_style)
        self.modify_button.pack(side="left", padx=10)

        # Scale Entry Frame
        self.scale_frame = tk.Frame(self.root, bg="#2E2E2E")
        self.scale_frame.pack(pady=10, padx=20, fill="x")
        tk.Label(self.scale_frame, text="Scale (px/um):", bg="#2E2E2E", fg="#ffffff", font=("Arial", 12)).pack(side="left")
        self.scale_input = tk.Entry(self.scale_frame, bg="#444444", fg="#ffffff", insertbackground="white")
        self.scale_input.pack(side="left", padx=10)

        # Image Display Frame
        self.image_frame = tk.Frame(self.root, bg="#1c1c1c")
        self.image_frame.pack(expand=True, fill="both", padx=20, pady=20)

        self.image_label = tk.Label(self.image_frame, bg="#000000")  # Black background for the image area
        self.image_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Bind resize event
        self.root.bind("<Configure>", self.on_resize)

        self.current_image = None

        # Bind window resize event
        self.root.bind("<Configure>", self.on_resize)

        # Variables
        self.current_image = None
        self.image_path = None
        self.weights = None
        self.results_folder = None


    def open_weights(self):
        """Open and select weights file."""
        weights_path = filedialog.askopenfilename(title="Select Weights File")
        if weights_path:
            print(f"Selected weights file: {weights_path}")
            self.weights = Path(weights_path)

    def set_scale(self):
        """Set the scale input value."""
        self.image_scale = self.scale_input.get()
        print(f"The scale is {self.image_scale}")

    def open_results_folder(self):
        """Open and select the results folder."""
        results_folder = filedialog.askdirectory(title="Select Results Folder")
        if results_folder:
            print(f"Selected results folder: {results_folder}")
            self.results_folder = Path(results_folder)

    def open_image(self):
        """Open an image and display it."""
        file_path = filedialog.askopenfilename(title="Select Image File")
        if file_path:
            self.image_path = Path(file_path)
            if file_path.endswith((".tiff", ".tif")):
                messagebox.showinfo("Info", f"Image path: {file_path} but it will not display.")
                # Load image for processing (assuming normalize_PIL is a function)
                self.current_image = open_tiff_image(str(file_path))
            else:
                self.current_image = Image.open(file_path)

            self.aspect_ratio = self.current_image.width / self.current_image.height
            self.display_image()

    def display_image(self):
        """Display the image in the image label."""
        if self.current_image:
            img_width, img_height = self.resize_image(self.image_label.winfo_width(), self.image_label.winfo_height())
            resized_image = self.current_image.copy()
            resized_image.thumbnail((img_width, img_height))

            self.tk_image = ImageTk.PhotoImage(resized_image)
            self.image_label.config(image=self.tk_image)
            self.image_label.image = self.tk_image  # Keep reference

    def resize_image(self, width, height):
        """Resize the image while maintaining its aspect ratio."""
        max_width = width - 20  # Subtract padding
        max_height = height - 20  # Subtract padding
        scale = min(max_width / self.current_image.width, max_height / self.current_image.height)

        return int(self.current_image.width * scale), int(self.current_image.height * scale)

    def on_resize(self, event):
        """Handle window resize event."""
        if event.widget == self.root:
            self.root.after(100, self.display_image)

    def modify_image(self):
        """Apply image modification based on selected option."""
        self.set_scale()
        if not self.image_path:
            messagebox.showerror("Error", "Please open an image first.")
            return

        modification = self.modification.get()
        print(f"Applying {modification} modification")

        try:
            # Assuming image_segmenter and gen_seg are defined elsewhere
            self.SAM_ob = image_segmenter(self.weights, self.results_folder, modification, SCALE=int(self.image_scale))
            final_image = self.SAM_ob.gen_seg(self.image_path)
            self.current_image = Image.fromarray(final_image)
            self.display_image()
            messagebox.showinfo("Success", f"Applied {modification} to the image")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply modification: {str(e)}")

    def run(self):
        """Run the main application loop."""
        self.root.mainloop()


if __name__ == "__main__":
    ImageEditor().run()