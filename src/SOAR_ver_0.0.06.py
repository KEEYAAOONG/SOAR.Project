"""
##########################################################


[SOAR ver_0.0.06]


Fertilization: 2025.04.08
Birth: 2025.04.11
Made by: Keeyaaoong, (Chat-GPT; Monday) 1-NOAH / 2-MAUND / 3-LEN / 4- GRIM


##########################################################
"""

"""
<Version Log> #############################################


2025.04.08 ver 0.0.00 - 카메라(저화질) + 마이크 + 감정 계산(쾌/불쾌) + 기억 저장/불러오기 + 날짜/시간별 백업 + 실시간 감정 모니터링
2025.04.09 ver 0.0.01 - 종료 개선 + Clustering 추가 + Clustering 시각화 + 감정 Neutral 범위 추가 + 소리민감도 수정(30)
2025.04.09 ver 0.0.02 - 감각 본체 + 실시간 시각화 스레드 분리 (Threaded version)
2025.04.09 ver 0.0.03 - 소리 패턴 분석 (FFT) + 이름 패턴 학습(경험 기반) + Memory AutoSave(30 min) + TkAgg + 플롯 스레드 개선(종료 안정화)
2025.04.09 ver 0.0.04 - RAM 문제; 데이터 메모리(memory append) + 이름 인식 표현 + Clustering 시각화 축 변경
2025.04.10 ver 0.0.05 - Memory-Tag 추가 + 자동 메모리 정리 시스템 + 쾌/불쾌 판단(소리변화율) + Clustering 시각화 개선
2025.04.11 ver 0.0.06 - 텍스트 상호작용(입력 전후 10 sec 소리/영상 통합 기억 저장) + data format 수정 + Lagacy 호환성 유지 + Background sound + 텍스트 입력 창 분리 + 자동저장(하루단위)


"""



import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import pyaudio
import numpy as np
import time
import json
import os
import threading
import queue
import tkinter as tk
from tkinter import simpledialog
from collections import deque
from sklearn.cluster import KMeans


# -----------------------------------------------
# 기본 설정
# -----------------------------------------------


CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120
MIC_DEVICE_INDEX = 1    # USB 마이크


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


EMOTION_THRESHOLD = 0.7


MEMORY_DIR = "Updated_Memory"
memory = []
learning_mode = False
is_learning_session_active = False  # 전역 변수로 추가
NAME_PATTERN_MEMORY = []
data_points = deque(maxlen=1000)
brightness_history = deque(maxlen=5)
volume_history = deque(maxlen=5)
emotion_history = deque(maxlen=100)
important_event_history = deque(maxlen=100)
plot_queue = queue.Queue()
exit_flag = threading.Event()
SAVE_INTERVAL = 1800
last_save_time = time.time()
recent_audio = deque(maxlen=int(RATE / CHUNK * 10))  # 10초 분량 오디오 버퍼
recent_visual = deque(maxlen=300)  # 10초 분량 비주얼 프레임 (약 30fps 가정)
background_memory = deque(maxlen=300)  # 최근 30초 정도 소리 저장


# -----------------------------------------------
# 기억 불러오기 및 저장하기
# -----------------------------------------------


def load_memory():
    global memory
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
    files = sorted([f for f in os.listdir(MEMORY_DIR) if f.startswith('memory_') and f.endswith('.json')])
    total_memories = []
    for filename in files:
        memory_path = os.path.join(MEMORY_DIR, filename)
        with open(memory_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'memories' in data:
                total_memories.extend(data['memories'])
            else:
                total_memories.extend(data)
    memory = total_memories
    print(f"[Memory Loaded] {len(memory)} memories restored from {len(files)} files.")




def save_memory():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
    today = time.strftime("%Y.%m.%d")
    memory_path = os.path.join(MEMORY_DIR, f"memory_{today}.json")
    data_to_save = {
        "version": "0.0.06",
        "last_updated": today,
        "memories": memory
    }
    with open(memory_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"[Memory Saved] {memory_path}")




# -----------------------------------------------
# 감정 및 기억 처리
# -----------------------------------------------


def calculate_emotion(brightness_change, volume_rate_of_change):
    emotion_score = 0.0
    if brightness_change > 30 or volume_rate_of_change > 0.3:
        if brightness_change > 50 or volume_rate_of_change > 0.5:
            emotion_score = -1.0
        else:
            emotion_score = 1.0
    return max(-1.0, min(1.0, emotion_score))


def maybe_store_memory(input_summary, emotion_score):
    if abs(emotion_score) >= EMOTION_THRESHOLD:
        memory.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": "Pleasure" if emotion_score > 0 else ("Displeasure" if emotion_score < 0 else "Neutral"),
            "strength": abs(emotion_score),
            "content": input_summary
        })


# -----------------------------------------------
# 이름 인식 처리
# -----------------------------------------------


def update_background_memory(current_fft):
    background_memory.append(current_fft)


def is_background_noise(current_fft):
    if len(background_memory) < 10:
        return False  # 데이터 부족하면 무조건 이벤트로 간주
    average_background = np.mean(background_memory, axis=0)
    similarity = np.linalg.norm(current_fft - average_background)
    return similarity < 8000  # Threshold (실험해서 조정 가능)


def update_name_memory(current_fft):
    global NAME_PATTERN_MEMORY
    if len(NAME_PATTERN_MEMORY) < 5:
        NAME_PATTERN_MEMORY.append(current_fft)
    else:
        similarity = np.mean([np.linalg.norm(current_fft - pattern) for pattern in NAME_PATTERN_MEMORY])
        if similarity < 10000:
            print("[SOAR]: Familiar sound detected (possible name recognition)")
            important_event_history.append(1)
            return
    important_event_history.append(0)




# -----------------------------------------------
# 학습 이벤트 함수 (텍스트 입력 + 오디오/비주얼 통합 저장)
# -----------------------------------------------


def capture_learning_event():
    global is_learning_session_active
    if is_learning_session_active:
        print("[Learning] 입력 세션이 이미 활성화되어 있습니다. 기다려주세요.")
        return


    is_learning_session_active = True  # 세션 시작
    root = tk.Tk()
    root.withdraw()


   


# -----------------------------------------------
# 플롯 스레드
# -----------------------------------------------


def plot_thread_func(q):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.show(block=False)


    while not exit_flag.is_set():
        try:
            while True:
                data = q.get_nowait()
                if data == 'EXIT':
                    exit_flag.set()
                    break


                emotion_history_local, data_points_local, important_event_local = data


                ax1.cla()
                for i, score in enumerate(emotion_history_local):
                    if i < len(important_event_local) and important_event_local[i] == 1:
                        color = 'red' if score < 0 else 'blue'
                    else:
                        color = 'gray'
                    ax1.scatter(i, score, color=color, s=10)
                ax1.set_ylim([-1.2, 1.2])
                ax1.set_title("SOAR Emotion Over Time")
                ax1.set_xlabel("Time (frames)")
                ax1.set_ylabel("Emotion Score")
                ax1.axhline(0, color='gray', linestyle='--')


                if len(data_points_local) >= 5:
                    X = np.array(data_points_local)
                    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
                    kmeans.fit(X)
                    labels = kmeans.labels_
                    centers = kmeans.cluster_centers_


                    ax2.cla()
                    for i in range(3):
                        cluster = X[labels == i]
                        ax2.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i}")
                    ax2.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centers')
                    ax2.set_title("SOAR Clustering")
                    ax2.set_xlabel("Brightness Change")
                    ax2.set_ylabel("Dominant Frequency")
                    ax2.legend()


                plt.tight_layout()
                plt.pause(0.001)


        except queue.Empty:
            continue


    plt.close('all')


# -----------------------------------------------
# 메인 실행
# -----------------------------------------------


load_memory()


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


cv2.namedWindow('SOAR - First Sight', cv2.WINDOW_NORMAL)  # 창 크기 조절 가능
cv2.resizeWindow('SOAR - First Sight', 640, 480)  # 원하는 창 크기로 설정


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_DEVICE_INDEX, frames_per_buffer=CHUNK)


plot_thread = threading.Thread(target=plot_thread_func, args=(plot_queue,))
plot_thread.start()


print("SOAR 감각 활성화 완료. (q를 누르면 종료)")


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break


        frame = cv2.flip(frame, 1)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_history.append(brightness)


        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        recent_audio.append(audio_data)
        recent_visual.append(frame)
        volume = np.linalg.norm(audio_data) / CHUNK
        volume_history.append(volume)


        fft_audio = np.fft.fft(audio_data)
        fft_magnitude = np.abs(fft_audio)
        dominant_frequency = np.argmax(fft_magnitude)


        update_background_memory(fft_magnitude)


        if is_background_noise(fft_magnitude):
            important_event_history.append(0)  # 배경 소리는 무시
        else:
            update_name_memory(fft_magnitude)  # 진짜 이벤트(이름)일 때만 인식


        if len(brightness_history) >= 2 and len(volume_history) >= 3:
            brightness_change = abs(brightness_history[-1] - brightness_history[-2])
            previous_volume = volume_history[-2]
            if previous_volume != 0:
                volume_rate_of_change = abs((volume_history[-1] - previous_volume) / previous_volume)
            else:
                volume_rate_of_change = 0


            emotion_score = calculate_emotion(brightness_change, volume_rate_of_change)
            emotion_history.append(emotion_score)


            input_summary = f"Brightness Change: {brightness_change:.2f}, Volume Rate of Change: {volume_rate_of_change:.2f}"
            maybe_store_memory(input_summary, emotion_score)


            data_points.append([brightness_change, dominant_frequency])


            if len(data_points) % 5 == 0:
                plot_queue.put((list(emotion_history), list(data_points), list(important_event_history)))


        if time.time() - last_save_time > SAVE_INTERVAL:
            save_memory()
            last_save_time = time.time()
            print("[Auto-Save]: Memory saved.")


        # 💥 여기!! 수정된 입력창 관리
        if learning_mode and not is_learning_session_active:
            is_learning_session_active = True
            root = tk.Tk()
            root.withdraw()


            word = simpledialog.askstring("SOAR 학습", "SOAR에게 가르칠 단어를 입력하세요 (취소 누르면 종료)", parent=root)
            if word is not None:
                word = word.strip()
                if word != '':
                    if recent_audio:
                        audio_data = np.concatenate(list(recent_audio))
                        fft_data = np.abs(np.fft.fft(audio_data))
                        audio_pattern = fft_data.tolist()
                    else:
                        audio_pattern = []


                    visual_frames_encoded = []
                    for frame in list(recent_visual):
                        _, buffer = cv2.imencode('.jpg', frame)
                        visual_frames_encoded.append(buffer.tolist())


                    memory_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "label": word,
                        "audio_pattern": audio_pattern,
                        "visual_frames": visual_frames_encoded,
                        "memory_importance": 0.5
                    }
                    memory.append(memory_entry)
                    save_memory()
                    print(f"[Learned] '{word}' 단어와 관련된 패턴 저장 완료.")


            try:
                root.destroy()
            except:
                pass


            is_learning_session_active = False
            learning_mode = False


        # 👑 여기까지 입력 처리


        cv2.imshow('SOAR - First Sight', frame)


        key = cv2.waitKey(1) & 0xFF


        if key == ord('q'):
            break
        if key == ord('t'):
            learning_mode = True  # <-- 플래그 ON






except KeyboardInterrupt:
    print("\n종료 요청 감지.")


plot_queue.put('EXIT')
exit_flag.set()
plot_thread.join()


cap.release()
stream.stop_stream()
stream.close()
audio.terminate()
cv2.destroyAllWindows()
save_memory()


print("SOAR 감각 시스템 종료.")





