import asyncio
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from TTS.api import TTS
import uvicorn
import os

# --- TẢI MODEL (Chỉ 1 lần khi API khởi động) ---
# Tương tự như việc bạn tạo thư mục, việc này chạy 1 lần
print("Đang tải model TTS...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Đảm bảo bạn đã cài đặt: pip install TTS
# Thay "tts_models/multilingual/multi-dataset/xtts_v2" bằng model viXTTS của bạn nếu có
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# THAY ĐƯỜNG DẪN NÀY
SPEAKER_WAV_PATH = "path/to/your_voice_sample.wav" 
if not os.path.exists(SPEAKER_WAV_PATH):
    print(f"CẢNH BÁO: Không tìm thấy file speaker_wav tại: {SPEAKER_WAV_PATH}")
    # Bạn có thể dùng một file mẫu của TTS nếu chưa có file riêng
    # SPEAKER_WAV_PATH = "female.wav" # (Đây là file ví dụ, bạn cần có file thật)

print(f"Model đã tải xong trên {device}.")
# ----------------------------------------------

app = FastAPI()

# Định nghĩa dữ liệu đầu vào (thay vì UploadFile)
class TTSRequest(BaseModel):
    text: str

# --- Tác vụ PRODUCER (Tương đương Bước 2: Process của bạn) ---
async def run_tts_producer(queue: asyncio.Queue, text: str):
    """
    Chạy model AI để sinh audio.
    Thay vì ghi ra 'output/', nó sẽ 'put' (đẩy) vào queue (RAM buffer).
    """
    print("Producer: Bắt đầu sinh audio...")
    try:
        # Gọi hàm stream của thư viện TTS
        stream_chunks = tts.tts_stream(
            text=text,
            speaker_wav=SPEAKER_WAV_PATH,
            language="vi",
            stream_chunk_size=20, 
            stream_play_sync=True
        )
        
        # Đẩy từng mẩu audio (chunk) vào queue
        for chunk in stream_chunks:
            if chunk:
                await queue.put(chunk)
                
    except Exception as e:
        print(f"Producer Lỗi: {e}")
    finally:
        # Khi hoàn tất, 'put' None để báo hiệu cho Consumer
        print("Producer: Đã sinh xong. Gửi tín hiệu kết thúc.")
        await queue.put(None)

# --- Tác vụ CONSUMER (Tương đương Bước 3: Download của bạn) ---
async def stream_audio_consumer(queue: asyncio.Queue, producer_task: asyncio.Task):
    """
    Lấy audio từ queue (RAM buffer).
    Thay vì dùng FileResponse (đọc từ đĩa), nó 'yield' (gửi)
    trực tiếp chunk cho client.
    """
    print("Consumer: Bắt đầu chờ chunk từ queue...")
    try:
        while True:
            # Lấy chunk từ queue (bị 'treo' lại nếu queue rỗng)
            chunk = await queue.get()

            if chunk is None:
                # Producer báo đã xong
                print("Consumer: Nhận tín hiệu kết thúc.")
                break
            
            # Gửi (yield) mẩu audio về cho client
            yield chunk
            
            queue.task_done()

    except asyncio.CancelledError:
        # Client đã ngắt kết nối
        print("Consumer Bị Hủy: Client đã ngắt kết noi. Hủy producer...")
        producer_task.cancel() # Hủy tác vụ AI
        
    finally:
        print("Consumer: Dọn dẹp.")
        if not producer_task.done():
            producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


# --- ENDPOINT DUY NHẤT (Gộp cả 3 bước của bạn lại) ---
@app.post("/synthesize-realtime")
async def synthesize_realtime(request: TTSRequest):
    """
    Đây là Endpoint Real-time.
    Nó không dùng file, mà dùng queue (buffer trong RAM).
    """
    # 1. Tạo một queue (buffer) cho riêng request này
    queue = asyncio.Queue()

    # 2. Bắt đầu tác vụ Producer (chạy AI)
    # (Tương đương /process của bạn, nhưng chạy nền)
    producer_task = asyncio.create_task(
        run_tts_producer(queue, request.text)
    )

    # 3. Trả về StreamingResponse
    # (Tương đương /download của bạn, nhưng lấy từ queue)
    return StreamingResponse(
        stream_audio_consumer(queue, producer_task),
        media_type="audio/wav" # Trả về âm thanh wav
    )

if __name__ == "__main__":
    print("Khởi chạy Uvicorn server tại http://127.0.0.1:8000")
    print("Hãy thử gọi POST http://127.0.0.1:8000/synthesize-realtime")
    uvicorn.run(app, host="127.0.0.1", port=8000)
