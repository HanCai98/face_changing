import mediapipe as mp
import cv2


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

    
    def findFaceMesh(self, img, draw=False):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        face = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # if you want to draw the landmarks, set draw = True 
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                
                for lm in faceLms.landmark:
                    ih, iw, ic = img.shape
                    x,y = float(lm.x*iw), float(lm.y*ih)
                    face.append([x,y])

        cv2.destroyAllWindows()
        return img, face
    


