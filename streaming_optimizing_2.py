import asyncio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from TTS.api import TTS
import uvicorn
import os

# --- TẢI MODEL (Chỉ 1 lần khi API khởi động) ---
print("Đang tải model TTS...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
SPEAKER_WAV_PATH = "path/to/your_voice_sample.wav" # <-- THAY ĐƯỜNG DẪN NÀY
print(f"Model đã tải xong trên {device}.")
# ----------------------------------------------

app = FastAPI()

# --- Logic Producer & Consumer (Tái sử dụng từ trước) ---
# Producer: Giống hệt như trước
async def run_tts_producer(queue: asyncio.Queue, text_to_speak: str):
    """
    Chạy AI, sinh audio cho 1 CÂU và 'put' vào queue.
    """
    print(f"Producer: Bắt đầu sinh audio cho: '{text_to_speak}'")
    try:
        stream_chunks = tts.tts_stream(
            text=text_to_speak,
            speaker_wav=SPEAKER_WAV_PATH,
            language="vi",
            stream_chunk_size=20,
            stream_play_sync=True
        )
        for chunk in stream_chunks:
            if chunk:
                await queue.put(chunk)
    except Exception as e:
        print(f"Producer Lỗi: {e}")
    finally:
        print("Producer: Đã sinh xong 1 câu. Gửi tín hiệu kết thúc.")
        await queue.put(None) # Báo hiệu kết thúc 1 câu

# Consumer: THAY ĐỔI NHỎ
# Thay vì 'yield', nó sẽ 'send_bytes' qua WebSocket
async def stream_audio_consumer(queue: asyncio.Queue, producer_task: asyncio.Task, websocket: WebSocket):
    """
    Lấy audio từ queue và 'send' (gửi) về cho client qua WebSocket.
    """
    print("Consumer: Bắt đầu chờ audio chunk...")
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                print("Consumer: Nhận tín hiệu kết thúc (1 câu).")
                break
            
            # Gửi mẩu audio (dưới dạng bytes) về cho client
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

# --- ENDPOINT WEBSOCKET ---
@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket: Client đã kết nối.")
    
    # Đây là "Text Buffer" mà đồng nghiệp bạn nói
    text_buffer = ""
    # Các "giới hạn" (dấu hiệu kết thúc câu)
    sentence_terminators = [".", "?", "!", "...", ":", "\n", ","]
    
    try:
        while True:
            # 1. Nhận text chunk từ client (LLM)
            text_chunk = await websocket.receive_text()
            
            if text_chunk == "END_OF_STREAM":
                print("WebSocket: Nhận tín hiệu kết thúc từ LLM.")
                # Xử lý nốt phần text còn lại trong buffer nếu có
                if text_buffer.strip():
                    await trigger_tts_for_sentence(websocket, text_buffer)
                break
            
            # 2. Thêm text chunk vào buffer
            text_buffer += text_chunk
            
            # 3. Kiểm tra "giới hạn" (sentence boundary)
            # Tìm vị trí của dấu câu đầu tiên
            split_pos = -1
            for terminator in sentence_terminators:
                pos = text_buffer.find(terminator)
                if pos != -1:
                    # Tìm thấy 1 dấu câu, đánh dấu vị trí cắt
                    split_pos = pos + len(terminator)
                    break
            
            if split_pos != -1:
                # 4. Tìm thấy "giới hạn". Cắt câu ra để nói
                sentence_to_speak = text_buffer[:split_pos].strip()
                # Giữ lại phần còn lại trong buffer
                text_buffer = text_buffer[split_pos:] 
                
                if sentence_to_speak:
                    # 5. Kích hoạt TTS cho câu này
                    await trigger_tts_for_sentence(websocket, sentence_to_speak)

    except WebSocketDisconnect:
        print("WebSocket: Client đã ngắt kết nối.")
    except Exception as e:
        print(f"WebSocket Lỗi: {e}")
    finally:
        print("WebSocket: Đóng kết nối.")

async def trigger_tts_for_sentence(websocket: WebSocket, sentence: str):
    """
    Hàm trợ giúp: Kích hoạt cặp Producer/Consumer cho 1 câu.
    """
    # Gửi một tin nhắn text về client báo "Tôi bắt đầu nói câu này"
    await websocket.send_text(f"SPEAKING:{sentence}")
    
    queue = asyncio.Queue()
    producer_task = asyncio.create_task(run_tts_producer(queue, sentence))
    # Chờ cho đến khi Consumer gửi xong audio của câu này
    await stream_audio_consumer(queue, producer_task, websocket)
    
    # Gửi tin nhắn text báo "Tôi đã nói xong câu này"
    await websocket.send_text(f"IDLE:{sentence}")

# ---
if __name__ == "__main__":
    print("Khởi chạy Uvicorn server tại http://127.0.0.1:8000")
    print("Endpoint WebSocket đang chờ tại ws://127.0.0.1:8000/ws/tts")
    uvicorn.run(app, host="127.0.0.1", port=8000)
