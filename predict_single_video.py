import os
import sys
import io
import cv2
import numpy as np
import tensorflow as tf

# Windows 환경 한글 깨짐 및 인코딩 에러 방지
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


def process_and_predict_video(video_path, model, target_size=(36, 36), total_frames=45, target_fps=15, amp_factor=15):
    """
    단일 비디오 파일을 로드하여 메모리 상에서 즉각 전처리(얼굴크롭 + CbCr 증폭)를 수행하고
    모델 예측 입력을 생성한 뒤 판별 결과를 반환합니다.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"비디오를 열 수 없습니다: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or np.isnan(original_fps):
        original_fps = 30.0

    # 추출할 프레임들의 원본 비디오 내 인덱스 목록
    target_indices = [int(round((i / target_fps) * original_fps)) for i in range(total_frames)]

    frame_count, saved_count = 0, 0
    last_face_box = None
    processed_frames = []

    print("🎞️  1. 비디오 프레임 추출 및 실시간 얼굴 전처리 중...")
    
    while cap.isOpened() and saved_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        is_first_match = True
        result_final = None

        while saved_count < total_frames and target_indices[saved_count] == frame_count:
            if is_first_match:
                # 첫 번째 매칭일 때만 얼굴 추출 및 CbCr 증폭
                if last_face_box is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))

                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        last_face_box = faces[0]
                    else:
                        fh, fw = frame.shape[:2]
                        box_sz = min(fh, fw) // 2
                        last_face_box = (fw // 2 - box_sz // 2, fh // 2 - box_sz // 2, box_sz, box_sz)

                if last_face_box is not None:
                    x, y, w, h = last_face_box
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

                    face_crop = frame[y:y + h, x:x + w]
                    
                    # BGR -> YCbCr -> CbCr x15 증폭 -> RGB 변환
                    ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
                    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

                    cr_amp = np.clip((cr_ch.astype(np.float32) - 128.0) * amp_factor + 128.0, 0, 255).astype(np.uint8)
                    cb_amp = np.clip((cb_ch.astype(np.float32) - 128.0) * amp_factor + 128.0, 0, 255).astype(np.uint8)

                    ycrcb_amp = cv2.merge([y_ch, cr_amp, cb_amp])
                    result_rgb = cv2.cvtColor(ycrcb_amp, cv2.COLOR_YCrCb2RGB)  # 모델 입력을 위해 RGB로 타겟팅

                    # 36x36 리사이즈 고정
                    result_resized = cv2.resize(result_rgb, target_size, interpolation=cv2.INTER_AREA)
                    result_final = np.clip(result_resized, 0, 255).astype(np.float32)
                
                is_first_match = False

            if result_final is not None:
                processed_frames.append(result_final)

            saved_count += 1
            
        frame_count += 1

    cap.release()
    
    if len(processed_frames) < total_frames:
        # 비디오가 짧을 경우 마지막 프레임 패딩 복사
        while len(processed_frames) < total_frames:
            processed_frames.append(processed_frames[-1])
            
    # 2. DeepPhys 입력 텐서 생성
    appearance_inputs = []
    motion_inputs = []
    
    for t in range(1, total_frames):
        I_t = processed_frames[t]
        I_t1 = processed_frames[t-1]
        
        # Appearance Input: 정규화 & (C, H, W)
        app_img = I_t / 255.0
        app_img = np.transpose(app_img, (2, 0, 1))
        appearance_inputs.append(app_img)
        
        # Motion Input: 정규화 & (C, H, W)
        diff = I_t - I_t1
        denom = I_t + I_t1 + 1e-7
        motion_img = diff / denom
        motion_img = motion_img / (np.std(motion_img) + 1e-7)
        motion_img = np.transpose(motion_img, (2, 0, 1))
        motion_inputs.append(motion_img)
        
    return np.array(appearance_inputs), np.array(motion_inputs)

def main():
    if len(sys.argv) < 2:
        print("사용법: python predict_single_video.py <비디오파일_경로>")
        print("예시  : python predict_single_video.py my_face.mp4")
        return
        
    video_path = sys.argv[1]
    amp_factor = 15
    if len(sys.argv) >= 3:
        try:
            amp_factor = int(sys.argv[2])
        except ValueError:
            pass
            
    model_path = r"G:\내 드라이브\PPG_Dataset\DeepFakesON-Phys_CelebDF_V2.h5"
    
    # 1. 모델 로드
    print("🤖 2. 딥페이크 판별 모델 로드 중...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ 모델 로드 완료!")
    
    try:
        # 2. 전처리 수행
        app_in, mot_in = process_and_predict_video(video_path, model, amp_factor=amp_factor)
        
        # 3. 모델 추론
        print("🔮 3. 딥페이크 여부 예측 중...")
        predictions = model.predict([app_in, mot_in], verbose=0)
        mean_score = np.mean(predictions)
        
        # 4. 결과 판정 (0.5 미만이면 가짜)
        decision = "가짜 (FAKE) 🚨" if mean_score < 0.5 else "진짜 (REAL) ✅"
        
        print("\n" + "=" * 50)
        print(f"🎬  분석 동영상: {os.path.basename(video_path)}")
        print("-" * 50)
        print(f"📈  판별 스코어 (Score) : {mean_score:.4f}")
        print(f"⚖️  최종 판정 결과     : {decision}")
        print(f"  * 설명: 0에 가까울수록 가짜(Fake), 1에 가까울수록 진짜(Real)입니다.")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
