from fastapi import FastAPI, HTTPException, Request
from src.inference import InferenceEngine
from datetime import datetime, timedelta
import threading

app = FastAPI()

inference_engine = None
last_accessed = None
TIMEOUT = timedelta(minutes=5)  # set the timeout period
lock = threading.Lock()


def load_model():
    global inference_engine
    # Load the model
    inference_engine = InferenceEngine("/home/jthomas/Documents/AAI520/final-project/llama2-7b-ubuntu-GPTQ")
    print("Model loaded")


def unload_model():
    global inference_engine
    # Unload the model (if any)
    inference_engine = None
    print("Model unloaded")


def check_timeout():
    global last_accessed
    while True:
        with lock:
            if inference_engine is not None and last_accessed is not None and datetime.now() - last_accessed > TIMEOUT:
                unload_model()
        threading.Event().wait(60)


# Start the timeout check thread
threading.Thread(target=check_timeout, daemon=True).start()


@app.post("/generate/")
async def generate_text(request: Request):
    global last_accessed
    with lock:
        if inference_engine is None:
            load_model()
        last_accessed = datetime.now()
        data = await request.json()
        input_text = data.get("input_text", "")
        if not input_text:
            raise HTTPException(status_code=400, detail="input_text field is required")
        if inference_engine is not None:
            output_text = inference_engine.run_inference(input_text)  # type: ignore
        else:
            raise HTTPException(status_code=500, detail="Model could not be loaded")

    return {"generated_text": output_text}
