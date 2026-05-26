import cv2
import numpy as np

video_path = "elden_ring_boss_start.mp4"
cam = cv2.VideoCapture(video_path)
fps = cam.get(cv2.CAP_PROP_FPS)
currentframe = 0

image_paths = []

while True:
    ret, frame = cam.read()
    if not ret:
        break
    if currentframe % int(fps) == 0:
        name = f"{video_path.split('.')[0]}_{currentframe:04d}.jpg"
        cv2.imwrite(name, frame)
        image_paths.append(name)
    currentframe += 1

cam.release()
print(f"Extracted {len(image_paths)} frames")

image_paths = image_paths[4:]
print(f"Using {len(image_paths)} frames (skipped first 4): {image_paths}")
images = [cv2.imread(p) for p in image_paths]
print(f"Loaded {len(images)} images")

cols = 3
rows = (len(images) + cols - 1) // cols
cell_w, cell_h = 426, 240
resized = [cv2.resize(im, (cell_w, cell_h)) for im in images]

composite = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
for i, im in enumerate(resized):
    r, c = divmod(i, cols)
    composite[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = im

cv2.imwrite(f"{video_path.split('.')[0]}_composite.jpg", composite)
print(f"Composite saved: {composite.shape[1]}x{composite.shape[0]} ({rows}x{cols} grid)")
