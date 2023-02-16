from os import listdir
from os.path import join
from PIL import Image
import os
import time

# Start timer
startTime = time.time()

# Loads all the images 
def load_images(container_path='datasets', folders=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']):

    filenames, labels = [], []

    # Add all the image names to the filenames array
    for label, folder in enumerate(folders):
        folder_path = join(container_path, folder)
        images = [join(folder_path, d)
                    for d in sorted(listdir(folder_path))]
        labels.extend(len(images) * [label])
        filenames.extend(images)

    return filenames


# Load the images using the above function
print('Loading images...')
images = load_images()
print('Images Loaded.')

# Necessary imports for cv2
import cv2

# Read images with cv2
print('Reading images with cv2...')
images = {name: cv2.imread(name) for name in images}
print('Done reading images.')

# Necessary imports for mediapipe
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print('Drawing annotations...')

# Array for all the annotated images
annotated_images = []

# Run MediaPipe Hands.
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for name, image in images.items():
    # Convert the BGR image to RGB, flip the image around y-axis for correct 
    # handedness output and process it with MediaPipe Hands.
    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))


    if not results.multi_hand_landmarks:
      continue

    # Draw hand landmarks of each hand.
    # Commenting below out to save on runtime
    image_height, image_width, _ = image.shape
    annotated_image = cv2.flip(image.copy(), 1)
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    annotated_images.append(annotated_image)

print('Finished drawing annotations.')

# Creates a new directory for the annotated images
directory = 'annotated_images'
os.mkdir(directory)

print('Saving images into new folder...')

# Save the annotated images
for idx, image in enumerate(annotated_images):
    # Save the image to the path
    cv2.imwrite(f'annotated_images\\annotated_image_{idx}.jpg', image)

print('Images saved.')


print('Annotated images length: ')
print(len(annotated_images))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))