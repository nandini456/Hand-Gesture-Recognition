# Hand-Gesture-Recognition
 Hand gesture recognition and detection is a technology that 
identifies and interprets hand movements or poses captured through cameras or sensors. It 
involves detecting the hand's presence in an image or video frame and recognizing specific 
gestures, such as waving, pointing, or forming symbols.
The provided script implements a **real-time hand gesture recognition system** using Mediapipe 
for hand tracking and custom classifiers for gesture detection. The landmarks are processed 
into normalized and relative coordinate features to make them invariant to hand size and 
position. The **KeyPointClassifier** identifies static hand gestures based on these landmark 
features, while the **PointHistoryClassifier** predicts dynamic finger movements by analyzing 
the trajectory of specific points over time. The system uses a deque to maintain a history of 
landmark positions and gesture classifications, enabling robust recognition of both static 
and dynamic gestures. The program can log new data for model training, enhancing the 
classifier's performance. 
For hand gesture recognition application, Streamlit is 
used to build a web interface where users can configure camera settings and control the 
detection process through a sidebar with options like selecting the camera device, adjusting 
resolution, and setting detection confidence levels. 
