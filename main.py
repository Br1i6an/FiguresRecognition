# Importar  librerias
from ultralytics import YOLO
import cv2

# Lectura del modelo
model = YOLO("modeloYoshi.pt")

# Realizar la  VideoCaptura
cap = cv2.VideoCapture(0)

while True:
    # Lectura de fotogramas
    ret, frame = cap.read()

    # Lectura de resultados
    resultados = model.predict(frame, imgsz=640, conf=0.70)

    # Mostrar resultados
    anotaciones = resultados[0].plot()

    # Mostrar fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar la ventana
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()