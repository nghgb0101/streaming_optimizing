import asyncio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from TTS.api import TTS
import uvicorn
import time
import re

# --- CẤU HÌNH CÁC ĐIỀU KIỆN XẢ ---
# 1. Dấu câu: Các ký tự kết thúc (ngắt) một câu
SENTENCE_TERMINATORS = re.compile(r'[.?!,;:\n…]')

# 2. Số từ: Xả buffer nếu số từ vượt quá ngưỡng này
MAX_WORD_COUNT = 20

# 3. Thời gian chờ: Xả buffer nếu LLM im lặng quá lâu (tính bằng giây)
MAX_WAIT_TIME = 1 # 1s

# 4. Kết thúc: Tín hiệu báo LLM đã nói xong
END_OF_STREAM_SIGNAL = "END_OF_STREAM"

# Thời gian poll (chờ) text mới từ WebSocket (tính bằng giây)
# Đặt thấp để phản ứng nhanh
POLL_TIMEOUT = 0.1 # 100ms

# --- TẢI MODEL (Chỉ 1 lần khi API khởi động) ---
print("Đang tải model TTS...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    # THAY ĐỔI: Sử dụng model XTTS-2 của Coqui
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    # THAY ĐƯỜNG DẪN NÀY:
    SPEAKER_WAV_PATH = "path/to/your_voice_sample.wav" 
    print(f"Model đã tải xong trên {device}.")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    print("Vui lòng kiểm tra đường dẫn model và file speaker_wav.")
    exit()

# --- KHỞI TẠO APP ---
app = FastAPI()

# --- LOGIC STREAMING ÂM THANH (PRODUCER/CONSUMER) ---
# Nhiệm vụ: Chạy AI, sinh audio và 'put' vào queue
async def run_tts_producer(queue: asyncio.Queue, text_to_speak: str):
    print(f"Producer: Đang sinh audio cho: '{text_to_speak}'")
    try:
        stream_chunks = tts.tts_stream(
            text=text_to_speak,
            speaker_wav=SPEAKER_WAV_PATH,
            language="vi",
            stream_chunk_size=20, # Kích thước chunk nhỏ để giảm độ trễ
            stream_play_sync=True
        )
        for chunk in stream_chunks:
            if chunk:
                await queue.put(chunk)
    except Exception as e:
        print(f"Producer Lỗi: {e}")
    finally:
        await queue.put(None) # Báo hiệu kết thúc 1 câu

# Nhiệm vụ: Lấy audio từ queue và 'send' qua WebSocket
async def stream_audio_consumer(queue: asyncio.Queue, producer_task: asyncio.Task, websocket: WebSocket):
    print("Consumer: Bắt đầu chờ audio chunk...")
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            await websocket.send_bytes(chunk)
            queue.task_done()
    except asyncio.CancelledError:
        print("Consumer Bị Hủy: Client ngắt kết nối? Hủy producer...")
        producer_task.cancel()
    finally:
        # Dọn dẹp
        if not producer_task.done():
            producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass

# Nhiệm vụ: Gói gọn việc khởi chạy 1 cặp Producer/Consumer
async def trigger_tts_for_sentence(websocket: WebSocket, sentence: str):
    if not sentence.strip():
        return
        
    print(f"Trigger: Bắt đầu TTS cho: '{sentence}'")
    # Gửi tin nhắn text báo "Tôi bắt đầu nói câu này"
    await websocket.send_text(f"SPEAKING:{sentence}")
    
    queue = asyncio.Queue()
    producer_task = asyncio.create_task(run_tts_producer(queue, sentence))
    await stream_audio_consumer(queue, producer_task, websocket)
    
    # Gửi tin nhắn text báo "Tôi đã nói xong câu này"
    await websocket.send_text(f"IDLE:{sentence}")


# --- ENDPOINT WEBSOCKET TỐI ƯU (4 ĐIỀU KIỆN XẢ) ---
@app.websocket("/ws/tts-optimized")
async def websocket_tts_optimized_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket: Client đã kết nối (Tối ưu).")
    
    # === Khởi tạo trạng thái cho logic 4 điều kiện ===
    text_buffer = "" # Bộ đệm Text
    last_flush_time = time.time() # Thời điểm xả buffer lần cuối
    # -------------------------------------------------

    # Hàm trợ giúp (helper) để xả buffer
    async def flush_buffer(buffer: str):
        """Kích hoạt TTS cho buffer hiện tại và reset timer."""
        nonlocal last_flush_time, text_buffer
        text_to_flush = buffer.strip()
        if text_to_flush:
            # Chạy TTS song song (fire-and-forget)
            # Chúng ta KHÔNG 'await' ở đây để vòng lặp nhận text
            # có thể tiếp tục ngay lập tức.
            asyncio.create_task(trigger_tts_for_sentence(websocket, text_to_flush))
        
        last_flush_time = time.time()
        text_buffer = "" # Xóa buffer sau khi xả

    try:
        while True:
            try:
                # 1. CHỜ TEXT MỚI (VỚI TIMEOUT NGẮN)
                # Chờ text mới trong một khoảng thời gian ngắn (POLL_TIMEOUT)
                text_chunk = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=POLL_TIMEOUT
                )
                
                # === XỬ LÝ KHI NHẬN ĐƯỢC TEXT ===
                
                # ĐIỀU KIỆN 4: LLM KẾT THÚC
                if text_chunk == END_OF_STREAM_SIGNAL:
                    print("Điều kiện 4: LLM Kết thúc.")
                    await flush_buffer(text_buffer) # Xả nốt phần còn lại
                    break # Thoát vòng lặp
                
                # Thêm text mới vào buffer
                text_buffer += text_chunk
                
                # ĐIỀU KIỆN 1: DẤU CÂU
                # Tìm dấu câu trong buffer
                match = SENTENCE_TERMINATORS.search(text_buffer)
                if match:
                    print("Điều kiện 1: Dấu câu.")
                    # Lấy vị trí dấu câu để cắt
                    split_pos = match.end()
                    sentence_to_speak = text_buffer[:split_pos]
                    text_buffer = text_buffer[split_pos:] # Giữ lại phần còn lại
                    
                    await flush_buffer(sentence_to_speak)
                    continue # Quay lại vòng lặp, không cần check 2-3

                # ĐIỀU KIỆN 2: SỐ TỪ
                word_count = len(text_buffer.split())
                if word_count >= MAX_WORD_COUNT:
                    print("Điều kiện 2: Vượt quá số từ.")
                    await flush_buffer(text_buffer)
                    continue

            except asyncio.TimeoutError:
                # === XỬ LÝ KHI KHÔNG NHẬN ĐƯỢC TEXT (TIMEOUT) ===
                
                # Không có text mới nào trong 0.1s qua
                # Bây giờ kiểm tra xem đã đến lúc timeout (Điều kiện 3) chưa
                
                time_since_last_flush = time.time() - last_flush_time
                
                # ĐIỀU KIỆN 3: THỜI GIAN CHỜ (TIMEOUT)
                if text_buffer and (time_since_last_flush > MAX_WAIT_TIME):
                    print("Điều kiện 3: Timeout (Im lặng quá lâu).")
                    await flush_buffer(text_buffer)
                
                pass # Hết timeout, vòng lặp tiếp tục

    except WebSocketDisconnect:
        print("WebSocket: Client đã ngắt kết nối.")
    except Exception as e:
        print(f"WebSocket Lỗi: {e}")
    finally:
        print("WebSocket: Đóng kết nối.")

# ---
if __name__ == "__main__":
    print("Khởi chạy Uvicorn server tại http://127.0.0.1:8000")
    print("Endpoint WebSocket tối ưu đang chờ tại ws://127.0.0.1:8000/ws/tts-optimized")
    uvicorn.run(app, host="127.0.0.1", port=8000)
