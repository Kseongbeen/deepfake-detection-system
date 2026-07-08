import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import os
import tempfile
from PIL import Image

# 페이지 설정: 와이드 모드 적용하여 좌우 공간 활용
st.set_page_config(page_title="DeepFake Detection", layout="wide")

# 레이아웃 안정화 CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 1rem !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; margin-top: -20px; padding-bottom: 10px;'>🛡️ 딥페이크 변조 영상 탐지 시스템</h2>", unsafe_allow_html=True)
st.markdown("---")

# 1. 버전 호환성을 고려한 모델 재구성 함수
def build_compatible_model():
    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False, 
        weights=None, 
        input_shape=(380, 380, 3)
    )
    x = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = layers.Dropout(0.4, name='dropout')(x)
    x = layers.Dense(512, activation=None, name='dense')(x)
    output = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name='l2_norm')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    return model

@st.cache_resource
def load_safe_model():
    hdf5_path = 'triplet_effB4_ep06_BS28.hdf5'
    model = build_compatible_model()
    try:
        model.load_weights(hdf5_path, by_name=True, skip_mismatch=True)
        return model
    except Exception as e:
        st.error(f"모델 가중치 로드 중 오류 발생: {e}")
        return None

@st.cache_resource
def load_mtcnn_model():
    pb_path = 'mtcnn.pb'
    if not os.path.exists(pb_path): return None
    with open(pb_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())
    def mtcnn_process(img, min_size, factor, thresholds):
        with tf.device('/cpu:0'):
            prob, landmarks, box = tf.compat.v1.import_graph_def(
                graph_def,
                input_map={'input:0': img, 'min_size:0': min_size, 'thresholds:0': thresholds, 'factor:0': factor},
                return_elements=['prob:0', 'landmarks:0', 'box:0'],
                name=''
            )
        return box, prob, landmarks
    return tf.compat.v1.wrap_function(mtcnn_process, [
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[3], dtype=tf.float32)
    ])

def margin_face(img, box):
    w, h = box[3] - box[1], box[2] - box[0]
    w_m, h_m = int(h * 0.135), int(w * 0.10)
    x_min, y_min = max(box[1] - w_m, 0), max(box[0] - h_m, 0)
    x_max, y_max = min(box[3] + w_m, img.shape[1]), min(box[2] + h_m, img.shape[0])
    return img[y_min:y_max, x_min:x_max].copy()

def get_max_size_box(bbox):
    if len(bbox) == 0: return -1
    sizes = [(box[3]-box[1]) * (box[2]-box[0]) for box in bbox]
    return np.argmax(sizes)

@st.cache_resource
def load_all_models():
    feature_model = load_safe_model()
    svm_model_path = 'triplet_effB4_ep06_BS28.pkl'
    if not os.path.exists(svm_model_path): return feature_model, None
    with open(svm_model_path, 'rb') as f:
        svm_model = pickle.load(f)
    return feature_model, svm_model


# 레이아웃 분할: 좌측(입력 및 영상), 우측(결과 및 설명)
left_col, right_col = st.columns([1.2, 1])

with left_col:
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
        
        analyze_btn = st.button("🔍 영상 분석 시작", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("분석 중..."):
                mtcnn_model = load_mtcnn_model()
                feature_model, svm_model = load_all_models()
                
                if feature_model is None or mtcnn_model is None or svm_model is None:
                    st.error("모델 로드 실패")
                else:
                    cap = cv2.VideoCapture(tfile.name)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    target_fps, duration = 15, 3
                    max_analyze_frames = target_fps * duration
                    frame_interval = max(1, int(fps / target_fps))
                    
                    faces_list = []
                    prog = st.progress(0)
                    
                    analyze_cnt = 0
                    for i in range(0, min(total_frames, int(fps * duration)), frame_interval):
                        if analyze_cnt >= max_analyze_frames: break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if not ret: break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        bbox, _, _ = mtcnn_model(frame_rgb.astype(np.float32), tf.constant(40.0), tf.constant(0.709), tf.constant([0.6, 0.7, 0.8], dtype=tf.float32))
                        bbox = bbox.numpy()
                        
                        if len(bbox) < 1:
                            frame_rot = cv2.rotate(frame_rgb, cv2.ROTATE_180)
                            bbox, _, _ = mtcnn_model(frame_rot.astype(np.float32), tf.constant(40.0), tf.constant(0.709), tf.constant([0.6, 0.7, 0.8], dtype=tf.float32))
                            bbox = bbox.numpy()
                            frame_to_crop = frame_rot if len(bbox) > 0 else frame_rgb
                        else:
                            frame_to_crop = frame_rgb
                            
                        if len(bbox) > 0:
                            max_idx = get_max_size_box(bbox)
                            face = margin_face(frame_to_crop, bbox[max_idx].astype(int))
                            try:
                                faces_list.append(cv2.resize(face, (380, 380)))
                            except: pass
                        
                        analyze_cnt += 1
                        prog.progress(min(1.0, analyze_cnt / max_analyze_frames))
                    
                    cap.release()
                    
                    if len(faces_list) > 0:
                        X = np.array(faces_list, dtype=np.float32)
                        embeddings = feature_model.predict(X, batch_size=10)
                        fake_probs = svm_model.predict_proba(embeddings)[:, 1]
                        avg_fake_prob = np.mean(fake_probs)
                        st.session_state['result'] = (avg_fake_prob, len(faces_list))
                    else:
                        st.error("얼굴 검출 실패")

with right_col:
    if 'result' in st.session_state:
        avg_fake_prob, face_cnt = st.session_state['result']
        st.markdown(f"### 📊 분석 결과")
        if avg_fake_prob > 0.5:
            st.error(f"**판정: 딥페이크(변조) 의심**")
        else:
            st.success(f"**판정: 원본(진짜) 영상**")
        
        res_cols = st.columns(2)
        res_cols[0].metric("변조 확률", f"{avg_fake_prob*100:.1f}%")
        res_cols[1].metric("분석 프레임", f"{face_cnt}장")
        st.progress(float(avg_fake_prob))
        st.markdown("---")
    
    st.markdown("#### ⚙️ 시스템 원리")
    st.info("""
    - **안면 검출**: MTCNN을 통한 정밀 얼굴 추출
    - **특징 추출**: EfficientNet-B4 기반 임베딩
    - **학습 기법**: Triplet Loss로 변조 흔적 극대화
    - **최종 판별**: SVM을 통한 고정밀 이진 분류
    """)
    st.caption("초반 3초(45프레임)를 기반으로 분석을 수행합니다.")


