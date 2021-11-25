import cv2
import mediapipe as mp
import time
import json
import numpy as np
import skimage.io as io


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, refine = False, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks = refine
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine_landmarks,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    # if you want to draw the landmarks, set draw = True
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        face = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # draw landmarks 
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                
                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x,y])

                                          
                indices = np.array([10, 152, 234, 454, 159, 145, 33, 133, 386, 374, 263, 362, 1, 13], dtype=int)
                face = np.array(face)
                dest = face[indices]
                dest = np.array(dest, dtype="float32")
                
    
                    
                # put the label indices of mask in the src[]
                mask_image = cv2.imread('../masks/mask_1.png',  cv2. IMREAD_UNCHANGED)
                # mask_image = cv2.cvtColor(mask_image,cv2.COLOR_BGR2RGB)
                file = open('../conf/mask_1_indices.json')
                mask_1_indices = json.load(file)

                src = []
                for key, value in mask_1_indices.items():
                    src.append(value)
                src = np.array(dest, dtype="float32")

                # get the perspective transformation matrix
                h, M = cv2.findHomography(src, dest, cv2.RANSAC,5.0)
                print(M)

                # transformed masked image
                output_shape = mask_image.shape[:2]
                print(output_shape)
                img = cv2.warpPerspective(mask_image, M,(img.shape[1], img.shape[0]), None, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)
                # print(transformed_mask.shape) # (720, 1280, 3)


        cv2.destroyAllWindows()
        return img, face
    


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print( "No camera found or error opening camera; using a static image instead." )

    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, face = detector.findFaceMesh(img)

        if face:
            print(len(face))
        else:
            print("face losing")
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    # cap.release()

if __name__ == "__main__":
    main()