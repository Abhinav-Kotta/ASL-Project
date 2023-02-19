from os import listdir
from os.path import join
from PIL import Image
import os
import shutil
import time

# Start timer
startTime = time.time()

# Loads all the images
def load_images(container_path='archive/asl_alphabet_train/asl_alphabet_train', folders=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], size=100):

    filenames, labels = [], []

    # Add all the image names to the filenames array
    for label, folder in enumerate(folders):
        folder_path = join(container_path, folder)
        dest_folder = os.path.join('archive/annotated', folder)
        if not os.path.exists('archive/annotated'):
            os.mkdir('archive/annotated')
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)

        src_folder = os.path.join(container_path, folder)
        images = [join(folder_path, d)
                    for d in sorted(listdir(folder_path))]

        for image in images:
            print(image)
            shutil.copy(image, dest_folder)

        labels.extend(len(images) * [label])
        filenames.extend(images)

    return filenames

images = load_images()