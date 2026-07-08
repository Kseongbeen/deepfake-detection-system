import os
import cv2
import numpy as np
import tensorflow as tf

def load_and_preprocess_video(video_dir, target_size=(36, 36)):
    """
    G드라이브의 전처리 폴더 내 frame_0000.jpg ~ frame_0044.jpg를 읽어와 
    DeepFakesON-Phys 모델 입력 규격에 맞게 변환합니다.
    - Appearance: I(t) -> 크기 (3, 36, 36), 정규화
    - Motion: I(t) - I(t-1) -> 크기 (3, 36, 36), 정규화
    """
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
    if len(frame_files) == 0:
        raise FileNotFoundError(f"폴더에 JPG 프레임이 없습니다: {video_dir}")
        
    frames = []
    for f in frame_files:
        img_path = os.path.join(video_dir, f)
        # OpenCV 한글 경로 대응으로 np.fromfile 사용
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # BGR -> RGB 변환 후 36x36 리사이즈
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32))
        
    if len(frames) < 2:
        raise ValueError("프레임 개수가 너무 부족합니다. (최소 2프레임 필요)")

    appearance_inputs = []
    motion_inputs = []
    
    # DeepPhys 표준 전처리: Motion = (I_t - I_t-1) / (I_t + I_t-1)
    for t in range(1, len(frames)):
        I_t = frames[t]
        I_t1 = frames[t-1]
        
        # Appearance Input: I_t 정규화 (0~1 범위 또는 표준화)
        # 여기서는 표준적인 0~1 스케일링 후 채널-퍼스트 (3, 36, 36) 변환
        app_img = I_t / 255.0
        app_img = np.transpose(app_img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        appearance_inputs.append(app_img)
        
        # Motion Input: (I_t - I_t-1) / (I_t + I_t-1 + epsilon)
        # 신호 증폭 및 정규화를 위한 표준 수식
        diff = I_t - I_t1
        denom = I_t + I_t1 + 1e-7
        motion_img = diff / denom
        
        # 표준편차로 나누어 정규화 (표준 DeepPhys 규격)
        motion_img = motion_img / (np.std(motion_img) + 1e-7)
        motion_img = np.transpose(motion_img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        motion_inputs.append(motion_img)
        
    return np.array(appearance_inputs), np.array(motion_inputs)

def main():
    model_path = r"G:\내 드라이브\PPG_Dataset\DeepFakesON-Phys_CelebDF_V2.h5"
    dataset_dir = r"G:\내 드라이브\PPG_Dataset\Fake"
    
    # 1. 모델 로드
    print("🤖 딥페이크 판별 모델 로드 중...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ 모델 로드 성공!\n")
    
    # 2. 전처리 완료된 비디오 폴더 리스트 확보 (최대 30개)
    all_video_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d != 'audio_driven2']
    video_folders = all_video_folders[:30]
    
    if not video_folders:
        print("❌ 테스트할 전처리 비디오 폴더가 없습니다.")
        return
        
    print(f"📊 총 {len(all_video_folders)}개의 비디오 중 상위 {len(video_folders)}개 배치 테스트를 진행합니다.")
    print("=" * 70)
    print(f"{'번호':^6}|{'비디오 이름':^30}|{'평균 Score':^12}|{'판정 결과':^12}")
    print("=" * 70)
    
    results = []
    fake_count = 0
    
    for idx, vname in enumerate(video_folders):
        vdir = os.path.join(dataset_dir, vname, "CbCr_x15")
        try:
            # 데이터 로딩 및 변환
            app_in, mot_in = load_and_preprocess_video(vdir)
            
            # 모델 예측
            predictions = model.predict([app_in, mot_in], verbose=0)
            mean_score = np.mean(predictions)
            
            # 판정 (0.5 미만이면 FAKE)
            decision = "FAKE" if mean_score < 0.5 else "REAL"
            if decision == "FAKE":
                fake_count += 1
                
            print(f" {idx+1:^4} | {vname:<28} | {mean_score:^10.4f} | {decision:^10}")
            results.append((vname, mean_score, decision))
            
        except Exception as e:
            print(f" {idx+1:^4} | {vname:<28} | {'Error':^10} | {'실패':^10} ({str(e)})")
            
    print("=" * 70)
    print(f"🏁 [배치 추론 요약]")
    print(f"  - 총 테스트 비디오: {len(video_folders)}개")
    print(f"  - 가짜(FAKE) 판정 비디오: {fake_count}개")
    print(f"  - 딥페이크 탐지 정확도 (가짜 판정 비율): {fake_count / len(video_folders) * 100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
