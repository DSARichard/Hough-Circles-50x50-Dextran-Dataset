import cv2
import numpy as np
import os

# circle detection
def circle_detect(gray_image):
  # input image must have shape (720, 72)
  if(gray_image.shape != (720, 72)):
    raise ValueError("input image must have shape (720, 72)")
  
  # manipulate image
  sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  gray_image = cv2.filter2D(gray_image, -1, sharpening_kernel)
  
  # detect circles
  # determine top left corner, width, and height of circle's bounding box
  circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1.05, 5, param2 = 8.5, minRadius = 1, maxRadius = 7)
  radii = np.int_(np.round(circles[0, :, 2]))
  left = np.int_(np.round(circles[0, :, 0])) - radii
  top = np.int_(np.round(circles[0, :, 1])) - radii
  left, top = np.clip(left, 0, 72), np.clip(top, 0, 720)
  width, height = (np.int_(np.round(circles[0, :, 2]*2)),)*2
  return np.array([left, top, width, height])

# custom list of frames
# note that these should come frome the same video
vidcap = cv2.VideoCapture("dextran_50x50_all.mp4")
success, img = vidcap.read()
img = np.uint8(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY))

# create image frames folder
if(not os.path.exists("dextran_frames")):
  os.makedirs("dextran_frames")

# detect tube inner wall indices based on brightness
# tube width normalized to 72
img_brt = np.mean(img, axis = 0)
img_brt = np.where(img_brt <= 88)[0]
wall_ind = np.where(img_brt[1:] - img_brt[:-1] > 50)[0][1]
right_tube_wall = (img_brt[wall_ind] + img_brt[wall_ind + 1])//2 + 37
left_tube_wall = right_tube_wall - 72
img = img[:, left_tube_wall:right_tube_wall]
cv2.imwrite("dextran_frames/frame0000.jpg", img)



# create JSON file (COCO format)
info = {
  "description": "Hough Circles Dataset",
  "url": "https://github.com/DSARichard/Hough-Circles-50x50-Dextran-Dataset",
  "version": "0.1.1",
  "year": 2022,
  "contributor": "DSARichard",
  "date_created": "2022-5-23 21:00:00" # datetime.datetime.utcnow().isoformat(" ")[:-7]
}
licenses = [
  {
    "id": 0,
    "name": "50x50 second version 5.24.2021",
    "url": "https://drive.google.com/drive/u/2/folders/1dcjWY2O3WdwHFDYSY7jjgSA1ZHZmGT4M"
  }
]
categories = [
  {
    "id": 0,
    "name": "hough circle",
    "supercategory": "circle"
  }
]
images = [
]
annotations = [
]

# add images and annotations
count = 0
prev_bbox = 0
while(success):
  # detect circles and write JSON file
  bbox = circle_detect(img).T
  image = {
    "id": count,
    "width": 72,
    "height": 720,
    "file_name": "dextran_frames/frame" + str(count).zfill(4) + ".jpg",
    "license": 0,
    "flickr_url": None,
    "coco_url": None,
    "date_captured": None
  }
  images.append(image)
  for bbox_id in range(len(bbox)):
    annotation = {
      "id": prev_bbox + bbox_id,
      "image_id": count,
      "category_id": 0,
      "bbox": bbox[bbox_id].tolist(),
    }
    annotations.append(annotation)
  prev_bbox = bbox_id + 1
  
  # next frame
  success, img = vidcap.read()
  if(img is None):
    continue
  img = np.uint8(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY))
  img_brt = np.mean(img, axis = 0)
  img_brt = np.where(img_brt <= 88)[0]
  wall_ind = np.where(img_brt[1:] - img_brt[:-1] > 50)[0][1]
  right_tube_wall = (img_brt[wall_ind] + img_brt[wall_ind + 1])//2 + 37
  left_tube_wall = right_tube_wall - 72
  img = img[:, left_tube_wall:right_tube_wall]
  count += 1
  cv2.imwrite("dextran_frames/frame" + str(count).zfill(4) + ".jpg", img)
  # if(count > 0):
  #   break

# write json file
json_dict = f'''{{
  "info": {info},
  "licences": {licenses},
  "categories": {categories},
  "images": {images},
  "annotations": {annotations}
}}'''.replace("None", "null").replace("'", '"')
f = open("dextran_v03b_50x50_dataset.json", "wt")
f.write(json_dict)
f.close()
