
import os
import uuid
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="CN-KH AI PRO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

ROOT = Path("/tmp/cnkh-jobs")
ROOT.mkdir(parents=True, exist_ok=True)

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
jobs = {}
lock = threading.Lock()


def client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("សូមកំណត់ OPENAI_API_KEY ក្នុង Server")
    return OpenAI(api_key=key)


class TextInput(BaseModel):
    text: str
    source_lang: str = "Chinese"
    target_lang: str = "Khmer"


def update(jid, **kwargs):
    with lock:
        jobs[jid].update(kwargs)


def run_ffmpeg(args):
    result = subprocess.run(
        ["ffmpeg", "-y"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2500:])


def translate(text, source="Chinese", target="Khmer"):
    response = client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate {source} into natural {target}. "
                    "Preserve meaning, names and numbers. "
                    "Output only the translation."
                )
            },
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content or ""


def timestamp(seconds):
    ms = int(seconds * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def process_job(jid, source):
    work = ROOT / jid
    work.mkdir(parents=True, exist_ok=True)

    try:
        c = client()

        update(
            jid,
            status="processing",
            progress=5,
            message="កំពុងស្រង់សំឡេងវីដេអូ..."
        )

        wav = work / "original.wav"

        run_ffmpeg([
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000",
            str(wav)
        ])

        update(
            jid,
            progress=15,
            message="កំពុងសម្គាល់សំឡេងចិន..."
        )

        with wav.open("rb") as f:
            result = c.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="zh",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        segments = result.segments or []

        if not segments:
            raise RuntimeError("មិនរកឃើញសំឡេងនិយាយក្នុងវីដេអូ")

        update(
            jid,
            progress=25,
            message="កំពុងបកប្រែជាភាសាខ្មែរ..."
        )

        translated = []

        for i, seg in enumerate(segments):
            kh = translate(seg.text)
            translated.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": kh
            })

            update(
                jid,
                progress=25 + int(20 * (i + 1) / len(segments)),
                message=f"កំពុងបកប្រែ {i+1}/{len(segments)}"
            )

        # Create SRT
        srt = work / "subtitles.srt"
        with srt.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(translated, 1):
                f.write(
                    f"{i}\n"
                    f"{timestamp(seg['start'])} --> "
                    f"{timestamp(seg['end'])}\n"
                    f"{seg['text']}\n\n"
                )

        update(
            jid,
            progress=48,
            message="កំពុងបង្កើតសំឡេងខ្មែរ..."
        )

        voice_files = []

        for i, seg in enumerate(translated):
            voice = work / f"voice_{i:04}.mp3"

            speech = c.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=seg["text"],
                response_format="mp3"
            )
            speech.stream_to_file(voice)
            voice_files.append(voice)

            update(
                jid,
                progress=48 + int(25 * (i + 1) / len(translated)),
                message=f"កំពុងបង្កើតសំឡេង {i+1}/{len(translated)}"
            )

        # Mix translated voices at original timestamps
        update(
            jid,
            progress=75,
            message="កំពុងរៀបចំសំឡេងតាមពេលវេលា..."
        )

        inputs = []
        filters = []

        for voice in voice_files:
            inputs.extend(["-i", str(voice)])

        for i, seg in enumerate(translated):
            delay = int(seg["start"] * 1000)
            filters.append(
                f"[{i}:a]adelay={delay}|{delay},"
                f"aresample=44100[a{i}]"
            )

        labels = "".join(
            f"[a{i}]" for i in range(len(voice_files))
        )

        filters.append(
            f"{labels}amix=inputs={len(voice_files)}:"
            "duration=longest:normalize=0[out]"
        )

        final_audio = work / "khmer.mp3"

        run_ffmpeg(
            inputs + [
                "-filter_complex", ";".join(filters),
                "-map", "[out]",
                "-c:a", "libmp3lame",
                "-b:a", "128k",
                str(final_audio)
            ]
        )

        update(
            jid,
            progress=88,
            message="កំពុងបង្កើតវីដេអូ MP4..."
        )

        output = work / "CN-KH-Translated.mp4"

        # Burn Khmer subtitles into video.
        subtitle_path = str(srt.resolve()).replace("\\", "/")
        subtitle_path = subtitle_path.replace(":", "\\:")

        run_ffmpeg([
            "-i", str(source),
            "-i", str(final_audio),
            "-vf", f"subtitles='{subtitle_path}'",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(output)
        ])

        update(
            jid,
            status="completed",
            progress=100,
            message="បកប្រែរួចរាល់!",
            transcript="\n".join(
                x["text"] for x in translated
            ),
            video=str(output),
            audio=str(final_audio),
            subtitle=str(srt)
        )

    except Exception as e:
        update(
            jid,
            status="failed",
            progress=0,
            message="ការបកប្រែបរាជ័យ",
            error=str(e)
        )


@app.get("/")
def home():
    return {
        "service": "CN-KH AI PRO",
        "status": "online",
        "features": ["MP4", "MP3", "SRT"]
    }


@app.post("/translate")
def translate_text(body: TextInput):
    try:
        return {
            "translation": translate(
                body.text,
                body.source_lang,
                body.target_lang
            )
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "សូមជ្រើសរើសវីដេអូ")

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "សូមផ្ញើឯកសារវីដេអូ")

    jid = uuid.uuid4().hex
    suffix = Path(file.filename).suffix or ".mp4"
    dest = ROOT / (jid + suffix)

    size = 0

    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_MB * 1024 * 1024:
                    raise HTTPException(413, "វីដេអូធំពេក")

                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    jobs[jid] = {
        "status": "queued",
        "progress": 2,
        "message": "បានទទួលវីដេអូ",
        "transcript": ""
    }

    threading.Thread(
        target=process_job,
        args=(jid, dest),
        daemon=True
    ).start()

    return {"id": jid}


@app.get("/jobs/{jid}")
def job_status(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "រកមិនឃើញការងារ")

    return {
        k: v for k, v in job.items()
        if k not in ("video", "audio", "subtitle")
    }


@app.get("/jobs/{jid}/video")
def job_video(jid: str):
    job = jobs.get(jid)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "វីដេអូមិនទាន់រួចរាល់")

    return FileResponse(
        job["video"],
        media_type="video/mp4",
        filename="CN-KH-Translated.mp4"
    )


@app.get("/jobs/{jid}/audio")
def job_audio(jid: str):
    job = jobs.get(jid)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "សំឡេងមិនទាន់រួចរាល់")

    return FileResponse(
        job["audio"],
        media_type="audio/mpeg",
        filename="CN-KH-Audio.mp3"
    )


@app.get("/jobs/{jid}/subtitle")
def job_subtitle(jid: str):
    job = jobs.get(jid)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Subtitle មិនទាន់រួចរាល់")

    return FileResponse(
        job["subtitle"],
        media_type="text/plain",
        filename="CN-KH-Subtitles.srt"
    )
    
