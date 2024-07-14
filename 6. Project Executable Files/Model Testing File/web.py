import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd  # Ensure pandas is imported

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Function to extract keypoints
def extract_keypoints(results):
    if results is None:
        return np.zeros(1864)  # Ensure the length matches what the model expects

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    additional_features = np.zeros(1864 - len(np.concatenate([pose, face, lh, rh])))
    
    return np.concatenate([pose, face, lh, rh, additional_features])

# Load the model
model_path = 'C:/Users/lenovo/Desktop/Body_Language_Decoder/model/body_language.pkl'

@st.cache(allow_output_mutation=True)
def load_model():
    with open('C:/Users/lenovo/Desktop/Body_Language_Decoder/model/body_language.pkl', 'rb') as f:
        return pickle.load(f)

# Function to visualize probabilities
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f'{actions[num]}: {prob:.2f}', (0, 85+ num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Streamlit App
st.write("""# Body Language Detection""")
mp_drawing = mp.solutions.drawing_utils
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

if run:
    model = load_model()
    actions = ['happy', 'sad', 'victorious', 'fight']
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0)]

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            ret, frame = camera.read()
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make Detections
            results = holistic.process(image)
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS, 
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            try:
                if results.pose_landmarks and results.face_landmarks:
                    keypoints = extract_keypoints(results)
                    
                    # Ensure X has the correct number of features
                    X = pd.DataFrame([keypoints])
                    
                    # Make Detections
                    body_language_prob = model.predict_proba(X)[0]
                    body_language_class = np.argmax(body_language_prob)
                    
                    # Debug: Print the raw probabilities and the predicted class
                    st.write(f'Probabilities: {body_language_prob}')
                    st.write(f'Predicted Class: {actions[body_language_class]}')
                    
                    # Visualize probabilities
                    image = prob_viz(body_language_prob, actions, image, colors)
                    
                    # Visualize the predicted class
                    cv2.putText(image, f'Class: {actions[body_language_class]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    st.write("No landmarks detected.")
                
            except Exception as e:
                st.write(f"Error: {e}")
                pass
            
            FRAME_WINDOW.image(image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
    camera.release()
    cv2.destroyAllWindows()
else:
    st.write('Stopped')
