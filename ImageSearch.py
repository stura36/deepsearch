import os
import tkinter as tk
from tkinter import filedialog

from PIL import ImageTk, Image
from sentence_transformers import util
import torch
from torchvision.io import read_image

from model_definition import retrieval_model



class ImageApp:
    def __init__(self, model_dict_path="model_weights.pth"):
        # creates main tkinter app
        app = tk.Tk()
        # sets app title
        app.title("Semantic Image Search")
        # stores tkinter app object
        self.app = app
        # adds all gui compomnents
        self.create_gui_comp()
        # loads model into the RAM
        self.model = self.load_model(model_dict_path)
        return

    def load_model(self, model_dict_path):
        # loads model from the given path into the mock model object
        retrieval_model.load_state_dict(torch.load(model_dict_path), strict=False)
        # sets model to evaulation mode
        retrieval_model.eval()

        return retrieval_model

    def start_app(self):
        """
        Starts the GUI application main loop.
        """
        self.app.mainloop()
        return

    def create_gui_comp(self):
        """
        Creates all the GUI elements that are placed into the main application window.
        returns - None
        """
        # Creates label and adds it to a main app window
        description_label = tk.Label(self.app, text="Enter description:")
        # Packs label
        description_label.pack()

        # Creates entry field for a text query and adds it to a main app window
        description_entry = tk.Entry(self.app)
        description_entry.pack()

        # Creates a button for choosing the folder which will be semanticaly searched.
        folder_path_button = tk.Button(
            self.app, text="File location", command=self.ask_for_dir
        )
        folder_path_button.pack()

        # Creates a button for calculating image embeddings
        compare_button = tk.Button(
            self.app,
            text="Compare Images",
            command=lambda: self.compare_images(),
        )
        compare_button.pack()

        # Creates a placeholder for the image display
        similar_image_label = tk.Label(self.app)
        similar_image_label.pack()

        # Creates a placeholder for the image name
        similar_image_name_label = tk.Label(self.app)
        similar_image_name_label.pack()

        # Stores all the created widgets references into the self object for further usage
        self.description_label = description_label
        self.description_entry = description_entry
        self.compare_button = compare_button
        self.similiar_image_label = similar_image_label
        self.similar_image_name_label = similar_image_name_label
        self.folder_path_button = folder_path_button

        return

    def ask_for_dir(self):
        # Asks user to choose a directory
        self.folder_path = filedialog.askdirectory()
        return

    def compare_images(self):
        # retrive folder path info
        folder_path = self.folder_path
        # retrive description query from Entry field
        description = self.description_entry.get()

        # infere text embeddings with model
        txt_emb = self.model.inference_txt(description)

        # Initialize variable for storing closest image
        most_similar_image = None

        # Initialize variable for storing highest similarity
        highest_similarity = -1

        # TODO: Iterate only over all image files
        # Iterate over all images in the folder(folder_path) and subfolders
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create image path from file name and root path
                image_path = os.path.join(root, file)


                with torch.no_grad():
                    image = read_image(image_path)
                    image_embedding = self.model.inference_image(image)

                #Calculate cosine similarity between text and image embeddings
                similarity = util.cos_sim(txt_emb, image_embedding).item()

                #Compare highest similarity with current similarity
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_image = image_path

        # Display the most similar image and its name on the screen
        self.image_path = image_path
        self.display_image(most_similar_image)

        return

    def display_image(self, image_path):
        """
        Functions for displaying the image in the main Tkinter window.
        Arguments:
            image_path - Path to the image

        returns - None
        """
        self.displayed_image = ImageTk.PhotoImage(Image.open(image_path))
        self.similiar_image_label.configure(image=self.displayed_image)
        self.similar_image_name_label.configure(text=os.path.basename(image_path))
        return

    def read_text(self):
        text = self.description_entry.get()
        return text

    def process_txt(self):
        return

if __name__ == "__main__":
    app = ImageApp()
    app.start_app()
