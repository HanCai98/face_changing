import cv2
import time
from FaceMeshModule import FaceMeshDetector  
from transformation import transform


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print( "No camera found or error opening camera; using a static image instead." )

    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, face = detector.findFaceMesh(img)

        if len(face) != 0:
            img = transform(face, img, '../masks/mask_1.png', '../conf/mask_1_indices.json')
        else:
            print("face losing")
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        # press 'Esc' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()