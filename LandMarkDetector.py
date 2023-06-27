import sys, cv2, dlib, time
import numpy as np
import faceBlendCommon as fbc
import matplotlib.pyplot as plt
import skimage
import os

class LandMarkDetector:

  def __init__(self):
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    self.save_path_folder = 'out'

  def get_landmarks(self, img):

    points = fbc.getLandmarks(self.detector, self.predictor, img)

    return np.array(points)

  def crop_square(self, img, face_points, final_dim=(299, 299)):

    minx, maxx, miny, maxy = int(np.min(face_points[:, 1])), int(np.max(face_points[:, 1])), int(np.min(face_points[:, 0])), int(np.max(face_points[:, 0]))
    sides = [maxy - miny, maxx - minx]
    cropped_img = img[minx:maxx, miny:maxy]

    top, bottom, left, right = 0, 0, 0, 0
    make_border = True
    if sides[0] > sides[1]:
        diff = sides[0] - sides[1]
        top = int(diff / 2)
        bottom = int(diff / 2) if diff % 2 == 0 else int(diff / 2) + 1
    elif sides[1] > sides[0]:
        diff = sides[1] - sides[0]
        right = int(diff / 2)
        left = int(diff / 2) if diff % 2 == 0 else int(diff / 2) + 1
    else:
        make_border = False

    if make_border:
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        new_im = cv2.resize(new_im, final_dim)
    else:
        new_im = cv2.resize(cropped_img, final_dim)

    return new_im

  def crop_face(self, img, face_points, img_name):

    face_borders = cv2.convexHull(face_points)[:, 0, :]
    Y, X = skimage.draw.polygon(face_borders[:, 1], face_borders[:, 0])
    blacked_img = np.zeros(img.shape, dtype=np.uint8)
    blacked_img[Y, X] = img[Y, X]
    cropped_img = self.crop_square(blacked_img, face_borders, final_dim=(299, 299))
    cv2.imwrite(os.path.join(self.save_path_folder, f'{img_name}.png'), cropped_img)

    return cropped_img


  def video2croppedImages(self, video_path, name_prefix=''):

    cap = cv2.VideoCapture(video_path)
    ret = True
    failed = 0
    i = 0
    while ret:
      i += 1
      try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        points = self.get_landmarks(frame)
        _ = self.crop_face(frame, points, img_name=name_prefix + f'_{str(i)}')

      except Exception as e:
        failed += 1
        print('FAILED #', failed)
        #print(e)

        if failed > 10:
          break

    return


  def empty_dir(self, dir_adr):

    for f in os.listdir(dir_adr):
      os.remove(os.path.join(dir_adr, f))


if __name__ == '__main__':

    detector = LandMarkDetector()
    video_adr = os.path.join('videos', 'id31_0005.mp4')
    # detector.video2croppedImages(video_path=video_adr, name_prefix='test')
    detector.empty_dir('out')