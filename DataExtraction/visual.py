import cv2
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import Sequential, Model
import cv2



def extract_color_features(frame, bins=64):
    """
    Extract color histogram features from a frame.

    :param frame: The frame from the video (as a NumPy array).
    :param bins: Number of bins for the histogram.
    :return: Normalized color histogram feature.
    """
    # Calculate the histogram for each color channel
    hist_features = []
    for i in range(3):  # Assuming frame is in BGR format
        hist = cv2.calcHist([frame], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    
    return np.array(hist_features)

def compute_optical_flow(prev_frame, curr_frame):
    """
    Compute optical flow between two frames.

    :param prev_frame: The previous frame in the video.
    :param curr_frame: The current frame in the video.
    :return: Optical flow magnitude and angle.
    """
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Compute magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize and flatten
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX).flatten()
    angle = angle.flatten()

    return magnitude, angle

# model = VGG16(weights='imagenet', include_top=False)

# Load the weights from the downloaded file
base_model = VGG16(weights=None, include_top=False)
weights_path = 'vgg16/vgg16_weights.h5' # Replace with the actual path
base_model.load_weights(weights_path)

# Create a new Sequential model and add the VGG16 base model
model = Sequential()
model.add(base_model)

# reduce 7x7x512 to 512 using GlobalAveragePooling2D
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Create a new model
model = Model(inputs=base_model.input, outputs=x)

def extract_visual_features(frames):
    features = []
    for frame in frames:
        if frame is not None:
            img = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
            img = img_to_array(img)              # Convert to array
            img = np.expand_dims(img, axis=0)    # Add batch dimension
            img = preprocess_input(img)          # Preprocess for VGG16
            feature = model.predict(img)
            features.append(feature.flatten())

    return features



def integrate_features(frames,vgg_features=None):
    """
    Integrate color, motion, and VGG features for a list of frames.

    :param frames: List of frames from the video.
    :param bins: Number of bins for color histogram.
    :return: Integrated feature vector for each frame.
    """
    integrated_features = []
    if(vgg_features is None):
        vgg_features = extract_visual_features(frames) 
    
    integrated_features.append([extract_color_features(frames[0]),vgg_features[0]])
    
    for i in range(1, len(frames)):
        color_features = extract_color_features(frames[i])
        # magnitude, angle = compute_optical_flow(frames[i-1], frames[i])


        # combined_features = np.concatenate([color_features, vgg_features[i]])
        # integrated_features.append([color_features, magnitude, angle, vgg_features[i]])
        
        integrated_features.append([color_features,vgg_features[i]])
    return integrated_features

