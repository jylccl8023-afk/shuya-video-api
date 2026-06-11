"""
小精灵舒雅 - 画中画视频合成 API 服务
=====================================
接收：背景视频URL + 配音音频URL + 张嘴PNG + 闭嘴PNG
输出：合成后的MP4视频URL

部署：pip install fastapi uvicorn librosa pillow pydub python-multipart
      需要系统安装 ffmpeg

接口：POST /compose
"""

import os
import uuid
import tempfile
import subprocess
import logging
from pathlib import Path

import librosa
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shuya_video_api")

app = FastAPI(title="舒雅视频合成API", version="1.0.0")

# ============ 配置区 ============
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output_videos")
STATIC_BASE_URL = os.environ.get("STATIC_BASE_URL", "http://localhost:8000/videos")
FPS = 24
PNGTUBER_SCALE = 0.45  # 舒雅占画面比例
PNGTUBER_POSITION = "bottom-right"  # 画中画位置
AMPLITUDE_THRESHOLD = 0.02  # 张嘴振幅阈值，越小越灵敏
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ComposeRequest(BaseModel):
    background_video_url: str
    audio_url: str
    open_mouth_png_url: str
    closed_mouth_png_url: str
    fps: int = FPS
    pngtuber_scale: float = PNGTUBER_SCALE
    pngtuber_position: str = PNGTUBER_POSITION
    amplitude_threshold: float = AMPLITUDE_THRESHOLD


def download_file(url: str, suffix: str) -> str:
    """下载文件到临时目录"""
    logger.info(f"下载文件: {url}")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    for chunk in resp.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    logger.info(f"下载完成: {tmp.name}")
    return tmp.name


def analyze_audio_mouth(audio_path: str, fps: int, threshold: float) -> list[bool]:
    """分析音频振幅，返回每帧是否张嘴的布尔列表"""
    logger.info("分析音频振幅...")
    y, sr = librosa.load(audio_path, sr=None)
    samples_per_frame = int(sr / fps)
    n_frames = int(len(y) / samples_per_frame)
    
    mouth_states = []
    for i in range(n_frames):
        frame_audio = y[i * samples_per_frame : (i + 1) * samples_per_frame]
        rms = np.sqrt(np.mean(frame_audio ** 2))
        mouth_states.append(rms > threshold)
    
    # 补齐到音频实际时长对应的帧数
    total_duration = len(y) / sr
    total_frames = int(total_duration * fps)
    while len(mouth_states) < total_frames:
        mouth_states.append(False)
    
    logger.info(f"音频分析完成，共 {len(mouth_states)} 帧，张嘴帧数: {sum(mouth_states)}")
    return mouth_states


def compose_pngtuber_frame(
    bg_frame_path: str,
    open_img: Image.Image,
    closed_img: Image.Image,
    is_mouth_open: bool,
    scale: float,
    position: str,
    output_path: str,
):
    """将PNGTuber贴到背景帧上"""
    bg = Image.open(bg_frame_path).convert("RGBA")
    bg_w, bg_h = bg.size
    
    overlay = open_img if is_mouth_open else closed_img
    
    # 缩放PNGTuber
    overlay_h = int(bg_h * scale)
    overlay_w = int(overlay.width * (overlay_h / overlay.height))
    overlay = overlay.resize((overlay_w, overlay_h), Image.LANCZOS)
    
    # 计算位置
    margin = int(bg_h * 0.03)
    if position == "bottom-right":
        x = bg_w - overlay_w - margin
        y = bg_h - overlay_h - margin
    elif position == "bottom-left":
        x = margin
        y = bg_h - overlay_h - margin
    elif position == "top-right":
        x = bg_w - overlay_w - margin
        y = margin
    elif position == "top-left":
        x = margin
        y = margin
    else:
        x = bg_w - overlay_w - margin
        y = bg_h - overlay_h - margin
    
    bg.paste(overlay, (x, y), overlay)
    bg.convert("RGB").save(output_path, "JPEG", quality=95)


def extract_background_frames(video_path: str, output_dir: str, fps: int) -> tuple[str, int]:
    """从背景视频提取帧，返回帧目录和总帧数"""
    logger.info("提取背景视频帧...")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(output_dir, "frame_%06d.jpg")
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    logger.info(f"提取完成，共 {len(frames)} 帧")
    return output_dir, len(frames)


def get_video_duration(video_path: str) -> float:
    """获取视频时长"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


@app.post("/compose")
async def compose_video(req: ComposeRequest):
    """
    核心合成接口
    请求体示例：
    {
        "background_video_url": "https://xxx/bg.mp4",
        "audio_url": "https://xxx/audio.mp3",
        "open_mouth_png_url": "https://xxx/open.png",
        "closed_mouth_png_url": "https://xxx/closed.png"
    }
    """
    task_id = str(uuid.uuid4())[:8]
    work_dir = tempfile.mkdtemp(prefix=f"shuya_{task_id}_")
    logger.info(f"[{task_id}] 开始合成任务，工作目录: {work_dir}")

    try:
        # 1. 下载所有素材
        bg_video_path = download_file(req.background_video_url, ".mp4")
        audio_path = download_file(req.audio_url, ".mp3")
        open_png_path = download_file(req.open_mouth_png_url, ".png")
        closed_png_path = download_file(req.closed_mouth_png_url, ".png")

        # 2. 分析音频 → 口型序列
        mouth_states = analyze_audio_mouth(
            audio_path, req.fps, req.amplitude_threshold
        )

        # 3. 提取背景视频帧
        frames_dir = os.path.join(work_dir, "bg_frames")
        bg_frames_dir, bg_frame_count = extract_background_frames(
            bg_video_path, frames_dir, req.fps
        )

        # 4. 加载PNGTuber图片（只加载一次）
        open_img = Image.open(open_png_path).convert("RGBA")
        closed_img = Image.open(closed_png_path).convert("RGBA")

        # 5. 逐帧合成：背景帧 + PNGTuber画中画
        composed_dir = os.path.join(work_dir, "composed_frames")
        os.makedirs(composed_dir, exist_ok=True)

        total_frames = max(bg_frame_count, len(mouth_states))
        bg_frames = sorted(Path(bg_frames_dir).glob("frame_*.jpg"))

        for i in range(total_frames):
            # 如果背景帧不够，循环使用最后一帧
            bg_idx = min(i, len(bg_frames) - 1)
            bg_frame_path = str(bg_frames[bg_idx])
            
            is_mouth_open = mouth_states[i] if i < len(mouth_states) else False
            
            output_frame = os.path.join(composed_dir, f"frame_{i+1:06d}.jpg")
            compose_pngtuber_frame(
                bg_frame_path, open_img, closed_img,
                is_mouth_open,
                req.pngtuber_scale, req.pngtuber_position,
                output_frame
            )

        logger.info(f"[{task_id}] 帧合成完成，共 {total_frames} 帧")

        # 6. ffmpeg: 帧序列 + 音频 → 最终MP4
        output_filename = f"shuya_{task_id}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(req.fps),
            "-i", os.path.join(composed_dir, "frame_%06d.jpg"),
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path
        ]
        logger.info(f"[{task_id}] 合成视频...")
        subprocess.run(cmd, check=True, capture_output=True)
        
        video_url = f"{STATIC_BASE_URL}/{output_filename}"
        logger.info(f"[{task_id}] 合成完成! URL: {video_url}")

        return JSONResponse({
            "code": 0,
            "message": "success",
            "output_video_url": video_url,
            "task_id": task_id,
            "total_frames": total_frames,
        })

    except subprocess.CalledProcessError as e:
        logger.error(f"[{task_id}] ffmpeg错误: {e.stderr.decode() if e.stderr else str(e)}")
        raise HTTPException(status_code=500, detail=f"视频合成失败: {str(e)}")
    except Exception as e:
        logger.error(f"[{task_id}] 合成异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"合成失败: {str(e)}")
    finally:
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            for f in [bg_video_path, audio_path, open_png_path, closed_png_path]:
                try:
                    os.unlink(f)
                except:
                    pass
        except:
            pass


@app.get("/health")
async def health():
    return {"status": "ok"}


# 静态文件服务（生产环境建议用 nginx 替代）
from fastapi.staticfiles import StaticFiles
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
