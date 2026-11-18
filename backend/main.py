
import os, uuid, subprocess, threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="CN-KH AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

ROOT = Path("/tmp/cnkh-jobs")
ROOT.mkdir(exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
jobs = {}

def client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Server មិនទាន់កំណត់ OPENAI_API_KEY")
    return OpenAI(api_key=key)

class TextInput(BaseModel):
    text: str

def translate(text):
    c = client()
    results = []
    for i in range(0, len(text), 3000):
        response = c.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Translate Chinese into natural Khmer. Preserve meaning, names and numbers. Output only Khmer."
                },
                {"role": "user", "content": text[i:i+3000]}
            ]
        )
        results.append(response.choices[0].message.content or "")
    return "\n".join(results)

@app.get("/")
def home():
    return {"service": "CN-KH AI", "status": "online"}

@app.post("/translate")
def translate_text(body: TextInput):
    try:
        return {"translation": translate(body.text)}
    except Exception as e:
        raise HTTPException(500, str(e))

def process_job(jid, source):
    work = ROOT / jid
    work.mkdir(exist_ok=True)

    try:
        jobs[jid].update(
            status="processing",
            progress=10,
            message="កំពុងស្រង់សំឡេង..."
        )

        wav = work / "audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000",
            str(wav)
        ], check=True, stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE)

        jobs[jid].update(
            progress=25,
            message="កំពុងបំបែកសំឡេងវីដេអូវែង..."
        )

        pattern = str(work / "part_%03d.mp3")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(wav),
            "-f", "segment", "-segment_time", "600",
            "-c:a", "libmp3lame", "-b:a", "64k",
            pattern
        ], check=True, stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE)

        parts = sorted(work.glob("part_*.mp3"))
        c = client()
        transcripts = []

        for n, part in enumerate(parts):
            jobs[jid].update(
                progress=25 + int(40 * n / max(1, len(parts))),
                message=f"កំពុងសម្គាល់សំឡេងចិន {n+1}/{len(parts)}"
            )
            with part.open("rb") as f:
                result = c.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="zh",
                    response_format="text"
                )
            transcripts.append(str(result))

        chinese = "\n".join(transcripts)

        jobs[jid].update(
            progress=68,
            message="កំពុងបកប្រែទៅភាសាខ្មែរ..."
        )
        khmer = translate(chinese)

        jobs[jid].update(
            progress=84,
            message="កំពុងបង្កើតសំឡេងខ្មែរ..."
        )

        # បំបែកអត្ថបទជាផ្នែកៗសម្រាប់ TTS
        chunks = [
            khmer[i:i+3500]
            for i in range(0, len(khmer), 3500)
        ]

        audio_files = []
        for i, chunk in enumerate(chunks):
            audio = work / f"voice_{i:03d}.mp3"
            speech = c.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=chunk,
                response_format="mp3"
            )
            speech.stream_to_file(audio)
            audio_files.append(audio)

        # ភ្ជាប់ MP3 ជាឯកសារតែមួយ
        list_file = work / "audio_list.txt"
        list_file.write_text(
            "".join(f"file '{p.name}'\n" for p in audio_files)
        )

        final_audio = work / "khmer.mp3"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy",
            str(final_audio)
        ], check=True, stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE)

        jobs[jid].update(
            status="completed",
            progress=100,
            message="រួចរាល់!",
            transcript=khmer,
            audio=str(final_audio)
        )

    except Exception as e:
        jobs[jid].update(
            status="failed",
            progress=0,
            message="ការបកប្រែបរាជ័យ",
            error=str(e)
        )
    finally:
        source.unlink(missing_ok=True)

@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "សូមផ្ញើឯកសារវីដេអូ")

    jid = uuid.uuid4().hex
    dest = ROOT / (jid + Path(file.filename or "video.mp4").suffix)
    size = 0

    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "វីដេអូធំពេក")
            out.write(chunk)

    jobs[jid] = {
        "status": "queued",
        "progress": 3,
        "message": "បានទទួលវីដេអូ"
    }

    threading.Thread(
        target=process_job,
        args=(jid, dest),
        daemon=True
    ).start()

    return {"id": jid}

@app.get("/jobs/{jid}")
def job_status(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "រកមិនឃើញការងារ")
    return {
        k: v for k, v in jobs[jid].items()
        if k != "audio"
    }

@app.get("/jobs/{jid}/audio")
def job_audio(jid: str):
    job = jobs.get(jid)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "សំឡេងមិនទាន់រួចរាល់")
    return FileResponse(
        job["audio"],
        media_type="audio/mpeg",
        filename="CN-KH-Khmer-Audio.mp3"
              )
  
