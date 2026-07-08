
import os
import sys

# Keras 3 (default in TF 2.16+) often fails with 'TensorFlowOpLayer' from Keras 2 models.
# Setting this env var tells TensorFlow to use the legacy tf_keras package.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

def check_imports():
    print("Checking imports...")
    try:
        import streamlit as st
        import cv2
        import numpy as np
        import tensorflow as tf
        import pickle
        import sklearn
        try:
            import tf_keras as keras
            print("Found tf_keras (Legacy Keras)")
        except ImportError:
            from tensorflow import keras
            print("Using tensorflow.keras")
        
        from keras.layers import DepthwiseConv2D
        import triplet_loss_dacon as triplet_loss
        print("✅ All imports successful.")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def check_model_loading():
    print("\nChecking model loading...")
    feature_model_path = 'triplet_effB4_ep06_BS28.hdf5'
    svm_model_path = 'triplet_effB4_ep06_BS28.pkl'
    
    import tensorflow as tf
    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras
        
    import triplet_loss_dacon as triplet_loss
    from keras.layers import DepthwiseConv2D
    import pickle

    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, **kwargs):
            if 'groups' in kwargs:
                kwargs.pop('groups')
            super().__init__(**kwargs)

    try:
        print(f"Loading feature model: {feature_model_path}")
        # Using keras.models.load_model instead of tf.keras.models.load_model 
        # to ensure we use the legacy one if TF_USE_LEGACY_KERAS is set.
        feature_model = keras.models.load_model(feature_model_path, custom_objects={
            'triplet_loss_adapted_from_tf': triplet_loss.triplet_loss_adapted_from_tf,
            'DepthwiseConv2D': CustomDepthwiseConv2D
        }, compile=False)
        print("✅ Feature model loaded successfully.")
        
        print(f"Loading SVM model: {svm_model_path}")
        with open(svm_model_path, 'rb') as f:
            svm_model = pickle.load(f)
        print("✅ SVM model loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_mtcnn_loading():
    print("\nChecking MTCNN loading...")
    pb_path = 'mtcnn.pb'
    import tensorflow as tf
    import os
    if not os.path.exists(pb_path):
        print(f"❌ MTCNN file not found: {pb_path}")
        return False
    
    try:
        with open(pb_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef.FromString(f.read())

        def mtcnn_process(img, min_size, factor, thresholds):
            with tf.device('/cpu:0'):
                prob, landmarks, box = tf.compat.v1.import_graph_def(
                    graph_def,
                    input_map={
                        'input:0': img,
                        'min_size:0': min_size,
                        'thresholds:0': thresholds,
                        'factor:0': factor
                    },
                    return_elements=['prob:0', 'landmarks:0', 'box:0'],
                    name=''
                )
            return box, prob, landmarks

        func = tf.compat.v1.wrap_function(mtcnn_process, [
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[3], dtype=tf.float32)
        ])
        print("✅ MTCNN loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ MTCNN loading failed: {e}")
        return False

def check_video_logic():
    print("\nChecking video logic...")
    video_path = '딥페이크_사람_예시_1.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")
    
    # app.py logic
    target_fps = 15
    duration = 3
    max_analyze_frames = target_fps * duration
    frame_interval = max(1, int(fps / target_fps))
    
    analyze_cnt = 0
    frames_to_read = []
    for i in range(0, min(total_frames, int(fps * duration)), frame_interval):
        if analyze_cnt >= max_analyze_frames: break
        frames_to_read.append(i)
        analyze_cnt += 1
    
    print(f"Logic: Target FPS {target_fps}, Duration {duration}s")
    print(f"Calculated frame indices to read: {frames_to_read}")
    print(f"Total frames to analyze: {len(frames_to_read)}")
    
    if len(frames_to_read) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_read[0])
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read first frame at index {frames_to_read[0]}")
        else:
            print(f"❌ Failed to read frame at index {frames_to_read[0]}")
            return False
    else:
        print("❌ No frames to analyze based on logic.")
        return False
    
    cap.release()
    return True

if __name__ == "__main__":
    s1 = check_imports()
    s2 = False
    if s1:
        s2 = check_model_loading()
    s_mtcnn = check_mtcnn_loading()
    s3 = check_video_logic()
    
    if s1 and s2 and s_mtcnn and s3:
        print("\n✨ All checks passed!")
    else:
        print("\n⚠️ Some checks failed.")
        sys.exit(1)
