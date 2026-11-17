"""Local UI and ordered WebSocket interaction. Bind to loopback for this prototype."""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .data import ROOT, load_rows, media_index
from .engine import Engine


@asynccontextmanager
async def lifespan(app):
    app.state.engine = Engine()
    app.state.rows = {split: load_rows(split) for split in ("train", "dev", "test")}
    app.state.media = media_index()
    # A small local prototype serializes inference to keep CPU latency predictable.
    app.state.inference_lock = asyncio.Lock()
    yield


app = FastAPI(title="Multimodal robot interaction", lifespan=lifespan)


@app.get("/")
async def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/app.js")
async def javascript():
    return FileResponse(ROOT / "web/app.js", media_type="application/javascript")


@app.get("/style.css")
async def stylesheet():
    return FileResponse(ROOT / "web/style.css", media_type="text/css")


@app.get("/api/dialogues")
async def dialogues(split: str = "dev"):
    if split not in app.state.rows:
        raise HTTPException(400, "Unknown split")
    return sorted(int(x) for x in app.state.rows[split].Dialogue_ID.unique())


@app.get("/api/turns")
async def turns(split: str = "dev", dialogue: int = 0):
    if split not in app.state.rows:
        raise HTTPException(400, "Unknown split")
    rows = app.state.rows[split]
    selected = rows[rows.Dialogue_ID == dialogue].sort_values("Utterance_ID")
    return [{"id": row.clip_id, "text": row.Utterance, "speaker": row.Speaker,
             "label": row.Emotion, "has_audio": row.clip_id in app.state.media[split]}
            for row in selected.itertuples()]


@app.get("/media/{split}/{clip_id}")
async def media(split: str, clip_id: str):
    path = app.state.media.get(split, {}).get(clip_id)
    if path is None:
        raise HTTPException(404, "Clip unavailable")
    return FileResponse(path, media_type="video/mp4")


class Turn(BaseModel):
    split: str = "dev"
    clip_id: str = Field(min_length=1, max_length=80)
    mode: str = "fusion"
    transcript: str | None = Field(default=None, max_length=4000)


@app.websocket("/ws")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                if len(raw) > 10000:
                    raise ValueError("Request is too large")
                turn = Turn.model_validate_json(raw)
                rows = app.state.rows.get(turn.split)
                if rows is None:
                    raise ValueError("Unknown split")
                selected = rows[rows.clip_id == turn.clip_id]
                if selected.empty:
                    raise ValueError("Unknown clip")
                transcript = turn.transcript if turn.transcript is not None else selected.iloc[0].Utterance
                path = app.state.media[turn.split].get(turn.clip_id)
                queued_at = asyncio.get_running_loop().time()
                async with app.state.inference_lock:
                    result = await asyncio.to_thread(app.state.engine.predict, transcript, path, turn.mode, turn.clip_id)
                result["server_elapsed_ms"] = round((asyncio.get_running_loop().time()-queued_at)*1000, 2)
                response = result.pop("response")
                await websocket.send_json({"type": "state", **result})
                # A deterministic template is chunked for transport. No token generator is used.
                for word in response.split(" "):
                    await websocket.send_json({"type": "response_chunk", "text": word + " "})
                await websocket.send_json({"type": "done", "response": response})
            except (ValueError, OSError, RuntimeError) as error:
                await websocket.send_json({"type": "error", "message": str(error)[:400]})
            except Exception:
                await websocket.send_json({"type": "error", "message": "Inference failed. Check local server logs and audio decoding."})
    except WebSocketDisconnect:
        return
