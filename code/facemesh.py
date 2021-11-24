import cv2
import mediapipe as mp
import time
 
 
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
 
 
# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
pTime = 0
 
with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
 
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
 
    # Draw the face mesh annotations on the image.
    # and print landmarks' id, x, y, z 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # Draw landmarks on the image.
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        # print id, x, y, z
        # time cost 
        """
        for id,lm in enumerate(face_landmarks.landmark):
            ih, iw, ic = image.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
            print(id, x,y,lm.z)
        """
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(image, f'FPS:{int(fps)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
 
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
      cv2.destroyAllWindows()
      cap.release()
      break