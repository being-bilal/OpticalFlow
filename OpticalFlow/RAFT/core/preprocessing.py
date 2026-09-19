import cv2 as cv
import numpy as np


def contrast_enhancement(img):
   lab_image = cv.cvtColor(img, cv.COLOR_BGR2LAB)
   L,A,B = cv.split(lab_image)
   # ClipLimit = Controls the enhancement limiting
   # TileGridSize = controls the size of grid 
   clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(8,8))
   cl = clahe.apply(L)
   limg = cv.merge((cl,A,B))
   enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
   return enhanced_img

def binary_image(img):
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   bw_img = cv.adaptiveThreshold(
         gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
         cv.THRESH_BINARY, 11, 4
      )
   return bw_img

if __name__ == "__main__":
   video = cv.VideoCapture('recording2.mp4')
   frame_count = 0
   while True:
      ret, frame = video.read()
      if not ret:
         break
      enhanced_frame = contrast_enhancement(frame)
      black_and_white_frame = binary_image(enhanced_frame)
      final_frame = black_and_white_frame
      frame_count += 1

      cv.imshow('frame', final_frame)
      cv.imshow('original', frame)
      if cv.waitKey(1) & 0xFF == ord('q'):
         break

   video.release()
   cv.destroyAllWindows()
   print(frame_count)

