import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('asl_model.h5')

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    img = cv2.resize(frame, (64, 64)) # same as training
    img = img / 255.0
    img = np.expand_dims(img, axis=0) # batch dimensions

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_letter = chr(65 + predicted_class) # 65 = ASCII for A

    cv2.putText(frame, f"Letter: {predicted_letter} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Translator', frame)

    # if q pressed quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()