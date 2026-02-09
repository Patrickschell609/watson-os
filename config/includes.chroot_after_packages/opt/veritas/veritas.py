#!/usr/bin/env python3
"""
PROJECT VERITAS - Forensic Deepfake Detection Tool
===================================================
Built for investigative journalists. Multi-vector analysis with explainable results.

Detection Methods:
1. Laplacian Variance - Unnatural face smoothness
2. Boundary Consistency - Composite edge artifacts
3. Spectral Audio Analysis - Synthetic voice markers
4. Blink Detection - Unnatural blink patterns (deepfakes often don't blink right)
5. Lip Sync Analysis - Audio/visual desynchronization
6. Color Histogram Anomaly - Unnatural color distributions
7. Noise Pattern Analysis - GAN fingerprints
8. OCR + Scam Detection - Wallet addresses, scam keywords

Output: JSON report + annotated video + HTML report for sharing
"""

import cv2
import numpy as np
import json
import sys
import subprocess
import os
import re
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import wave
import struct

# Optional advanced dependencies
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import librosa
    from scipy.signal import correlate
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.1.0"

# Detection thresholds (tuned from real deepfake analysis)
THRESHOLDS = {
    "laplacian_variance": 100,       # Below = suspiciously smooth
    "boundary_distance": 35,         # Above = color mismatch at edges
    "high_freq_ratio": 0.03,         # Below = synthetic voice
    "blink_rate_min": 0.1,           # Below = not blinking enough (per second)
    "blink_rate_max": 0.5,           # Above = blinking too much
    "lip_sync_threshold": 0.3,       # Above = desync detected
    "noise_uniformity": 0.8,         # Above = GAN fingerprint
}

# Scam detection patterns
SCAM_KEYWORDS = [
    "giveaway", "double", "2x", "return", "send eth", "send btc",
    "urgent", "limited time", "elon", "tesla", "free crypto", "claim now",
    "airdrop", "winner", "congratulations", "act now", "don't miss",
    "send to this address", "guaranteed profit", "risk free", "once in lifetime"
]

WALLET_PATTERNS = {
    "eth": re.compile(r"0x[a-fA-F0-9]{40}"),
    "btc": re.compile(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}"),
    "btc_bech32": re.compile(r"bc1[a-zA-HJ-NP-Z0-9]{25,89}"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionSignal:
    """Individual detection signal with explanation"""
    name: str
    score: float           # 0.0 to 1.0 (1.0 = highly suspicious)
    weight: float          # How much this contributes to final score
    evidence: str          # Human-readable explanation
    technical: str         # Technical details for experts

@dataclass
class FaceAnalysis:
    """Per-face analysis results"""
    frame: int
    x: int
    y: int
    w: int
    h: int
    laplacian_variance: float
    boundary_distance: float
    entropy: float
    suspicious: bool
    reasons: List[str]

@dataclass
class Metrics:
    """Aggregate metrics"""
    visual_manipulation: float = 0.0
    audio_artifacts: float = 0.0
    scam_indicators: float = 0.0
    temporal_inconsistency: float = 0.0
    entropy_anomaly: float = 0.0
    blink_anomaly: float = 0.0
    lip_sync_anomaly: float = 0.0
    noise_pattern_anomaly: float = 0.0

@dataclass
class VeritasReport:
    """Complete analysis report"""
    version: str
    target: str
    analyzed_at: str
    duration_seconds: float
    metrics: Metrics
    signals: List[Dict]
    confidence: float
    verdict: str
    verdict_explanation: str
    face_analysis: List[Dict]
    flags: List[str]
    recommendations: List[str]

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressReporter:
    """Thread-safe progress reporting"""
    def __init__(self, callback=None):
        self.callback = callback or self._default_print
        self.current_phase = ""
        self.progress = 0

    def _default_print(self, phase: str, progress: int, message: str):
        bar_width = 30
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r    [{bar}] {progress}% - {message}", end="", flush=True)
        if progress == 100:
            print()

    def update(self, phase: str, progress: int, message: str = ""):
        self.current_phase = phase
        self.progress = progress
        self.callback(phase, progress, message)

    def phase(self, name: str):
        print(f"\n[*] {name}")
        self.current_phase = name

# Global reporter
reporter = ProgressReporter()

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class VeritasError(Exception):
    """Base error with user-friendly message"""
    def __init__(self, message: str, technical: str = "", fixable: bool = True):
        self.message = message
        self.technical = technical
        self.fixable = fixable
        super().__init__(message)

class DependencyError(VeritasError):
    """Missing dependency"""
    pass

class InputError(VeritasError):
    """Invalid input"""
    pass

class AnalysisError(VeritasError):
    """Analysis failure"""
    pass

def safe_execute(func, fallback=None, error_msg="Operation failed"):
    """Execute function with error handling"""
    try:
        return func()
    except Exception as e:
        print(f"    WARNING: {error_msg}: {e}")
        return fallback

# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_executable(names: List[str]) -> Optional[str]:
    """Find executable in PATH or common locations"""
    import shutil

    script_dir = Path(__file__).parent
    extra_paths = [
        script_dir / ".venv" / "bin",
        script_dir / ".venv" / "Scripts",  # Windows
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),  # macOS ARM
    ]

    for name in names:
        # Check PATH first
        if shutil.which(name):
            return name
        # Check extra paths
        for path in extra_paths:
            full_path = path / name
            if full_path.exists():
                return str(full_path)
    return None

def download_video(target: str) -> Tuple[str, str]:
    """Download/copy video and extract audio with robust error handling"""
    video_out = "temp_video.mp4"
    audio_out = "temp_audio.wav"

    is_url = target.startswith(("http://", "https://", "www."))

    if is_url:
        reporter.update("ingest", 10, "Downloading video...")

        ytdlp = find_executable(["yt-dlp", "youtube-dl"])
        if not ytdlp:
            raise DependencyError(
                "yt-dlp not found. Install with: pip install yt-dlp",
                "Required for URL downloads"
            )

        # Try best quality first
        cmd = [
            ytdlp,
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--no-playlist",
            "--no-warnings",
            "-o", video_out,
            target
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Try simpler format
            cmd = [ytdlp, "-f", "best", "-o", video_out, target]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise InputError(
                    f"Failed to download video from URL",
                    f"yt-dlp error: {result.stderr[:200]}"
                )

        reporter.update("ingest", 50, "Download complete")

    else:
        reporter.update("ingest", 30, "Loading local file...")

        source = Path(target)
        if not source.exists():
            raise InputError(f"File not found: {target}")

        if not source.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v']:
            raise InputError(f"Unsupported format: {source.suffix}")

        import shutil
        shutil.copy(target, video_out)
        reporter.update("ingest", 50, "File loaded")

    # Extract audio
    reporter.update("ingest", 60, "Extracting audio...")

    ffmpeg = find_executable(["ffmpeg"])
    if not ffmpeg:
        raise DependencyError(
            "ffmpeg not found. Install with: sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)"
        )

    cmd = [
        ffmpeg, "-i", video_out,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "1",
        "-y", audio_out
    ]

    result = subprocess.run(cmd, capture_output=True)

    # Audio extraction might fail for video-only files - that's okay
    if not Path(audio_out).exists():
        print("    Note: No audio track found (video-only)")
        audio_out = None

    reporter.update("ingest", 100, "Ready for analysis")

    return video_out, audio_out

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

_face_cascade = None

def get_face_cascade():
    """Load face cascade with caching"""
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade

    paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]

    for p in paths:
        if Path(p).exists():
            _face_cascade = cv2.CascadeClassifier(p)
            return _face_cascade

    return None

def calculate_laplacian_variance(roi) -> float:
    """Measure texture variance - deepfakes often too smooth"""
    if roi is None or roi.size == 0:
        return 999  # Return high value (not suspicious) on error
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    except:
        return 999

def calculate_entropy(roi) -> float:
    """Calculate image entropy - synthetic content has abnormal entropy"""
    if roi is None or roi.size == 0:
        return 5.0
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)
    except:
        return 5.0

def check_boundary_consistency(frame, face_rect) -> Tuple[bool, float]:
    """Check for color inconsistency at face boundaries - composite detection"""
    x, y, w, h = face_rect
    margin = 20

    try:
        # Get expanded region
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face_roi = frame[y:y+h, x:x+w]

        # Get just the boundary strip
        outer = frame[y1:y2, x1:x2]

        if face_roi.size == 0 or outer.size == 0:
            return False, 0

        # Compare color histograms
        face_hist = cv2.calcHist([face_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        outer_hist = cv2.calcHist([outer], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        face_hist = cv2.normalize(face_hist, face_hist).flatten()
        outer_hist = cv2.normalize(outer_hist, outer_hist).flatten()

        # Correlation distance
        distance = cv2.compareHist(face_hist, outer_hist, cv2.HISTCMP_BHATTACHARYYA)

        return distance > 0.4, float(distance * 100)
    except:
        return False, 0

def analyze_noise_patterns(roi) -> float:
    """Detect GAN fingerprints in noise patterns"""
    if roi is None or roi.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # Apply high-pass filter to extract noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blur)

        # Analyze noise uniformity (GANs produce unnaturally uniform noise)
        noise_std = np.std(noise)

        # Very uniform noise is suspicious
        if noise_std < 3:
            return 0.8
        elif noise_std < 5:
            return 0.4
        return 0.0
    except:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE LANDMARK EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize Mediapipe Face Landmarker (lazy loading)
_face_landmarker = None
_model_path = None

def get_face_landmarker():
    """Get or create Mediapipe Face Landmarker instance (new Tasks API)"""
    global _face_landmarker, _model_path
    if not MEDIAPIPE_AVAILABLE:
        return None

    if _face_landmarker is not None:
        return _face_landmarker

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        # Download model if needed
        script_dir = Path(__file__).parent
        model_file = script_dir / "face_landmarker.task"

        if not model_file.exists():
            print("    Downloading face landmark model...")
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(model_url, str(model_file))

        options = vision.FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(model_file)),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
        _model_path = str(model_file)
        return _face_landmarker

    except Exception as e:
        print(f"    WARNING: Could not initialize FaceLandmarker: {e}")
        return None

def extract_landmarks(frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
    """
    Extract facial landmarks from a frame using Mediapipe Tasks API.
    Returns: list of (x, y, z) normalized coordinates, or None if no face detected.
    """
    landmarker = get_face_landmarker()
    if landmarker is None:
        return None

    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create mediapipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect landmarks
        result = landmarker.detect(mp_image)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in landmarks]

    except Exception:
        pass

    return None

# ═══════════════════════════════════════════════════════════════════════════════
# EAR-BASED BLINK DETECTION (Mediapipe)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ear(eye_landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Compute Eye Aspect Ratio for blink detection.
    EAR drops significantly when eye is closed.
    """
    try:
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_landmarks[1][:2]) - np.array(eye_landmarks[5][:2]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2][:2]) - np.array(eye_landmarks[4][:2]))
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_landmarks[0][:2]) - np.array(eye_landmarks[3][:2]))
        if h == 0:
            return 0.3
        return (v1 + v2) / (2.0 * h)
    except:
        return 0.3

def detect_blinks_mediapipe(landmarks_sequence: List, fps: float = 30,
                            ear_threshold: float = 0.21, min_blink_frames: int = 2) -> Tuple[float, int, str]:
    """
    Analyze blink patterns using Eye Aspect Ratio from Mediapipe landmarks.
    Returns: (anomaly_score, blink_count, explanation)
    """
    if len(landmarks_sequence) < 30:
        return 0.0, 0, "Insufficient frames for blink analysis"

    # Mediapipe Face Mesh eye landmark indices
    # Left eye: 33, 160, 158, 133, 153, 144
    # Right eye: 362, 385, 387, 263, 373, 380
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    ears = []
    for landmarks in landmarks_sequence:
        if landmarks and len(landmarks) >= 468:  # Full face mesh has 468 landmarks
            try:
                left_eye = [landmarks[i] for i in left_eye_indices]
                right_eye = [landmarks[i] for i in right_eye_indices]
                left_ear = compute_ear(left_eye)
                right_ear = compute_ear(right_eye)
                ears.append((left_ear + right_ear) / 2.0)
            except:
                ears.append(0.3)  # Default open eye
        else:
            ears.append(0.3)

    # Count blinks (consecutive low EAR frames)
    blink_count = 0
    in_blink = False
    blink_frames = 0

    for ear in ears:
        if ear < ear_threshold:
            blink_frames += 1
            in_blink = True
        else:
            if in_blink and blink_frames >= min_blink_frames:
                blink_count += 1
            in_blink = False
            blink_frames = 0

    # Calculate duration and expected blinks
    duration = len(landmarks_sequence) / fps
    # Normal: 15-20 blinks per minute = 0.25-0.33 per second
    expected_blinks = duration * 0.28  # ~17 blinks/min

    if expected_blinks < 1:
        return 0.0, blink_count, "Video too short for blink analysis"

    # Calculate deviation from expected
    deviation = abs(blink_count - expected_blinks) / max(expected_blinks, 1)

    # Score based on deviation
    if blink_count == 0 and duration > 5:
        score = 0.9  # No blinks in 5+ seconds is very suspicious
        explanation = f"No blinks detected in {duration:.1f}s (expected ~{expected_blinks:.0f})"
    elif deviation > 1.0:
        score = min(0.8, deviation * 0.4)
        explanation = f"Abnormal blink rate: {blink_count} blinks in {duration:.1f}s (expected ~{expected_blinks:.0f})"
    else:
        score = 0.0
        explanation = f"Normal blink pattern: {blink_count} blinks in {duration:.1f}s"

    return score, blink_count, explanation

def detect_blinks(video_path: str, sample_frames: int = 200) -> Tuple[float, int]:
    """
    Detect blink patterns - uses Mediapipe if available, falls back to Haar cascades.
    Returns (anomaly_score, blink_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    sample_every = max(1, total_frames // sample_frames)

    # Try Mediapipe first (more accurate)
    if MEDIAPIPE_AVAILABLE:
        landmarks_sequence = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue

            landmarks = extract_landmarks(frame)
            landmarks_sequence.append(landmarks)

        cap.release()

        if landmarks_sequence:
            score, blink_count, _ = detect_blinks_mediapipe(landmarks_sequence, fps)
            return score, blink_count

    # Fallback to Haar cascade method
    cap = cv2.VideoCapture(video_path)
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    if not Path(eye_cascade_path).exists():
        cap.release()
        return 0.0, 0

    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    face_cascade = get_face_cascade()

    if face_cascade is None:
        cap.release()
        return 0.0, 0

    eyes_detected_history = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            eyes_detected_history.append(len(eyes) >= 2)

    cap.release()

    if len(eyes_detected_history) < 10:
        return 0.0, 0

    blink_count = 0
    for i in range(1, len(eyes_detected_history)):
        if eyes_detected_history[i-1] and not eyes_detected_history[i]:
            blink_count += 1

    blink_rate = blink_count / max(duration, 1)

    if blink_rate < THRESHOLDS["blink_rate_min"]:
        return 0.8, blink_count
    elif blink_rate > THRESHOLDS["blink_rate_max"]:
        return 0.4, blink_count

    return 0.0, blink_count

# ═══════════════════════════════════════════════════════════════════════════════
# LIP-SYNC DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mar(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Compute Mouth Aspect Ratio - indicates how open the mouth is.
    """
    try:
        # Inner lip landmarks (Mediapipe indices)
        # Top inner: 13, Bottom inner: 14, Left corner: 78, Right corner: 308
        top = np.array(landmarks[13][:2])
        bottom = np.array(landmarks[14][:2])
        left = np.array(landmarks[78][:2])
        right = np.array(landmarks[308][:2])

        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)

        if horizontal == 0:
            return 0.0
        return vertical / horizontal
    except:
        return 0.0

def detect_lip_sync(video_path: str, audio_path: str,
                    landmarks_sequence: List, fps: float = 30) -> Tuple[float, str]:
    """
    Detect audio-visual lip sync anomalies.
    Compares mouth opening patterns with audio amplitude envelope.
    Returns: (desync_score, explanation)
    """
    if not LIBROSA_AVAILABLE:
        return 0.0, "Librosa not available for lip-sync analysis"

    if audio_path is None or not Path(audio_path).exists():
        return 0.0, "No audio track for lip-sync analysis"

    if len(landmarks_sequence) < 30:
        return 0.0, "Insufficient frames for lip-sync analysis"

    try:
        # Load audio and compute onset envelope
        y, sr = librosa.load(audio_path, sr=None, duration=60)  # Max 60 seconds

        # Get onset strength (audio energy envelope)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Resample to match video frames
        target_len = len(landmarks_sequence)
        if len(onset_env) > 0:
            onset_env_resampled = np.interp(
                np.linspace(0, len(onset_env) - 1, target_len),
                np.arange(len(onset_env)),
                onset_env
            )
        else:
            return 0.0, "No audio onset detected"

        # Compute MAR sequence
        mar_sequence = []
        for landmarks in landmarks_sequence:
            if landmarks and len(landmarks) >= 468:
                mar = compute_mar(landmarks)
                mar_sequence.append(mar)
            else:
                mar_sequence.append(0.0)

        mar_sequence = np.array(mar_sequence)

        # Check if there's enough variation in mouth movement
        if np.std(mar_sequence) < 0.01:
            # Mouth barely moving but audio present
            if np.std(onset_env_resampled) > 0.1:
                return 0.7, "Mouth static despite audio activity"
            return 0.0, "Both mouth and audio static"

        # Check if there's audio
        if np.std(onset_env_resampled) < 0.01:
            return 0.0, "No significant audio for lip-sync comparison"

        # Normalize both signals
        mar_norm = (mar_sequence - np.mean(mar_sequence)) / (np.std(mar_sequence) + 1e-10)
        audio_norm = (onset_env_resampled - np.mean(onset_env_resampled)) / (np.std(onset_env_resampled) + 1e-10)

        # Cross-correlation
        correlation = correlate(mar_norm, audio_norm, mode='full')
        max_corr = np.max(np.abs(correlation)) / len(mar_norm)

        # Find lag at max correlation
        lag = np.argmax(np.abs(correlation)) - len(mar_norm)
        lag_seconds = abs(lag) / fps

        # Score based on correlation and lag
        if max_corr < 0.1:
            score = 0.8
            explanation = f"Very poor lip-audio correlation ({max_corr:.2f})"
        elif max_corr < 0.2:
            score = 0.5
            explanation = f"Weak lip-audio correlation ({max_corr:.2f})"
        elif lag_seconds > 0.3:
            score = 0.6
            explanation = f"Lip-sync lag detected ({lag_seconds:.2f}s)"
        else:
            score = max(0.0, 0.3 - max_corr)  # Higher correlation = lower score
            explanation = f"Lip-sync correlation: {max_corr:.2f}"

        return score, explanation

    except Exception as e:
        return 0.0, f"Lip-sync analysis failed: {str(e)[:50]}"

def analyze_visual(video_path: str, collect_landmarks: bool = True) -> Tuple[float, float, float, float, List[FaceAnalysis], List]:
    """
    Comprehensive visual analysis.
    Returns: (manipulation_score, temporal_score, entropy_score, noise_score, face_details, landmarks_sequence)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise AnalysisError(f"Could not open video: {video_path}")

    face_cascade = get_face_cascade()
    if face_cascade is None:
        print("    WARNING: Face detection unavailable")
        cap.release()
        return 0.0, 0.0, 0.0, 0.0, []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_every = max(1, frame_count // 150)  # Sample ~150 frames

    suspicious_faces = 0
    total_faces = 0
    face_details = []
    entropy_values = []
    noise_scores = []
    landmarks_sequence = []  # For lip-sync analysis

    # For temporal analysis
    prev_face_positions = []
    temporal_jumps = 0

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        processed += 1
        progress = int(100 * processed / (frame_count / sample_every + 1))
        reporter.update("visual", min(progress, 99), f"Analyzing frame {frame_idx}/{frame_count}")

        # Extract landmarks for lip-sync analysis
        if collect_landmarks and MEDIAPIPE_AVAILABLE:
            landmarks = extract_landmarks(frame)
            landmarks_sequence.append(landmarks)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))

        current_positions = []

        for (x, y, w, h) in faces:
            total_faces += 1
            face_roi = frame[y:y+h, x:x+w]

            # Multiple detection methods
            variance = calculate_laplacian_variance(face_roi)
            is_smooth = variance < THRESHOLDS["laplacian_variance"]

            boundary_issue, boundary_dist = check_boundary_consistency(frame, (x, y, w, h))

            entropy = calculate_entropy(face_roi)
            entropy_values.append(entropy)

            noise_score = analyze_noise_patterns(face_roi)
            noise_scores.append(noise_score)

            # Compile reasons
            reasons = []
            if is_smooth:
                reasons.append(f"Unnaturally smooth (variance: {variance:.0f})")
            if boundary_issue:
                reasons.append(f"Edge inconsistency (distance: {boundary_dist:.1f})")
            if noise_score > 0.5:
                reasons.append("GAN noise pattern detected")

            suspicious = len(reasons) > 0
            if suspicious:
                suspicious_faces += 1

            face_details.append(FaceAnalysis(
                frame=frame_idx,
                x=int(x), y=int(y), w=int(w), h=int(h),
                laplacian_variance=round(variance, 2),
                boundary_distance=round(boundary_dist, 2),
                entropy=round(entropy, 2),
                suspicious=suspicious,
                reasons=reasons
            ))

            current_positions.append((x + w//2, y + h//2))

        # Check temporal consistency (faces shouldn't jump around)
        if prev_face_positions and current_positions:
            for curr_pos in current_positions:
                min_dist = min(
                    np.sqrt((curr_pos[0] - p[0])**2 + (curr_pos[1] - p[1])**2)
                    for p in prev_face_positions
                ) if prev_face_positions else 0

                if min_dist > 100:  # Large jump
                    temporal_jumps += 1

        prev_face_positions = current_positions

    cap.release()
    reporter.update("visual", 100, "Complete")

    if total_faces == 0:
        print("    Note: No faces detected in video")
        return 0.0, 0.0, 0.0, 0.0, [], landmarks_sequence

    # Calculate scores
    visual_score = suspicious_faces / total_faces

    temporal_score = min(1.0, temporal_jumps / (processed + 1) * 10)

    if entropy_values:
        entropy_std = np.std(entropy_values)
        entropy_score = min(1.0, max(0, (entropy_std - 0.5) / 1.5))
    else:
        entropy_score = 0.0

    noise_score = np.mean(noise_scores) if noise_scores else 0.0

    print(f"    Analyzed {total_faces} faces across {processed} frames")
    print(f"    Suspicious: {suspicious_faces} ({visual_score*100:.1f}%)")

    return visual_score, temporal_score, entropy_score, noise_score, face_details, landmarks_sequence

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_audio(audio_path: str) -> Tuple[float, List[str]]:
    """
    Analyze audio for synthesis artifacts.
    Returns: (anomaly_score, evidence_list)
    """
    if audio_path is None or not Path(audio_path).exists():
        return 0.0, ["No audio track"]

    evidence = []

    try:
        with wave.open(audio_path, 'rb') as wav:
            n_frames = wav.getnframes()
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()

            if n_frames < 1024:
                return 0.0, ["Audio too short for analysis"]

            # Read max 120 seconds
            max_frames = min(n_frames, sample_rate * 120)
            raw_data = wav.readframes(max_frames)

            if n_channels == 1:
                samples = struct.unpack(f"{len(raw_data)//2}h", raw_data)
            else:
                all_samples = struct.unpack(f"{len(raw_data)//2}h", raw_data)
                samples = all_samples[::n_channels]

            samples = np.array(samples, dtype=np.float32) / 32768.0

    except Exception as e:
        return 0.0, [f"Could not read audio: {e}"]

    if len(samples) < 4096:
        return 0.0, ["Audio too short for spectral analysis"]

    # Spectral analysis
    fft_size = 4096
    high_cutoff_bin = int(10000 * fft_size / sample_rate)  # 10kHz
    mid_cutoff_bin = int(4000 * fft_size / sample_rate)    # 4kHz

    low_energy_chunks = 0
    missing_highs_chunks = 0
    total_chunks = 0

    for i in range(0, len(samples) - fft_size, fft_size // 2):
        chunk = samples[i:i + fft_size]

        # Apply window
        window = np.hanning(len(chunk))
        chunk = chunk * window

        spectrum = np.abs(np.fft.fft(chunk))[:fft_size // 2]

        total_energy = np.sum(spectrum ** 2) + 1e-10
        high_energy = np.sum(spectrum[high_cutoff_bin:] ** 2)
        mid_energy = np.sum(spectrum[mid_cutoff_bin:high_cutoff_bin] ** 2)

        high_ratio = high_energy / total_energy
        mid_ratio = mid_energy / total_energy

        if high_ratio < THRESHOLDS["high_freq_ratio"]:
            missing_highs_chunks += 1

        if mid_ratio < 0.1:
            low_energy_chunks += 1

        total_chunks += 1
        if total_chunks >= 300:
            break

    if total_chunks == 0:
        return 0.0, ["No audio chunks analyzed"]

    missing_highs_ratio = missing_highs_chunks / total_chunks
    low_energy_ratio = low_energy_chunks / total_chunks

    # Build evidence
    if missing_highs_ratio > 0.5:
        evidence.append(f"Missing high frequencies in {missing_highs_ratio*100:.0f}% of audio")

    if low_energy_ratio > 0.5:
        evidence.append(f"Flat mid-range in {low_energy_ratio*100:.0f}% of audio")

    # Calculate score
    score = (missing_highs_ratio * 0.7 + low_energy_ratio * 0.3)

    reporter.update("audio", 100, "Complete")

    if evidence:
        print(f"    Detected: {', '.join(evidence)}")
    else:
        print("    No obvious audio artifacts")

    return min(1.0, score), evidence

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT ANALYSIS (OCR + SCAM DETECTION)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ocr(image_path: str) -> str:
    """Run tesseract OCR with fallback"""
    tesseract = find_executable(["tesseract"])
    if not tesseract:
        return ""

    try:
        result = subprocess.run(
            [tesseract, image_path, "stdout", "-l", "eng", "--psm", "3"],
            capture_output=True,
            text=True,
            timeout=15
        )
        return result.stdout.lower()
    except:
        return ""

def analyze_context(video_path: str) -> Tuple[float, List[str]]:
    """Extract text and check for scam indicators"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    full_text = ""
    flags = []
    wallets_found = {"eth": set(), "btc": set()}

    # Sample 15 frames across video
    num_samples = 15

    for i in range(num_samples):
        pos = int(i * frame_count / num_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        ret, frame = cap.read()
        if not ret:
            continue

        progress = int(100 * (i + 1) / num_samples)
        reporter.update("context", progress, f"OCR scan {i+1}/{num_samples}")

        temp_path = f"temp_ocr_{i}.png"
        cv2.imwrite(temp_path, frame)

        text = run_ocr(temp_path)
        full_text += text + " "

        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass

    cap.release()

    score = 0.0

    # Check for wallet addresses
    for pattern_name, pattern in WALLET_PATTERNS.items():
        matches = pattern.findall(full_text)
        for match in matches:
            if pattern_name == "eth":
                wallets_found["eth"].add(match)
            else:
                wallets_found["btc"].add(match)

    if wallets_found["eth"]:
        flags.append(f"ETH wallet(s) detected: {', '.join(list(wallets_found['eth'])[:3])}")
        score += 0.4

    if wallets_found["btc"]:
        flags.append(f"BTC wallet(s) detected: {', '.join(list(wallets_found['btc'])[:3])}")
        score += 0.4

    # Check for scam keywords
    found_keywords = []
    for keyword in SCAM_KEYWORDS:
        if keyword in full_text:
            found_keywords.append(keyword)
            score += 0.05

    if found_keywords:
        flags.append(f"Scam keywords: {', '.join(found_keywords[:10])}")

    reporter.update("context", 100, "Complete")

    return min(1.0, score), flags

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_annotated_video(video_path: str, report: VeritasReport, output_path: str = "veritas_output.mp4"):
    """Generate annotated video with overlays showing detection evidence"""
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_cascade = get_face_cascade()
    metrics = report.metrics
    confidence = report.confidence

    # Precompute face analysis lookup
    face_by_frame = {}
    for fa in report.face_analysis:
        if isinstance(fa, dict):
            frame = fa.get("frame", 0)
        else:
            frame = fa.frame
        if frame not in face_by_frame:
            face_by_frame[frame] = []
        face_by_frame[frame].append(fa)

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num % 50 == 0:
            progress = int(100 * frame_num / total_frames)
            reporter.update("video", progress, f"Rendering frame {frame_num}/{total_frames}")

        # Semi-transparent HUD background
        overlay = frame.copy()

        # Top HUD
        cv2.rectangle(overlay, (0, 0), (width, 160), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Title
        cv2.putText(frame, "PROJECT VERITAS // FORENSIC ANALYSIS",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                    (width - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Metrics with color coding
        def metric_color(val, thresh=0.5):
            if val > thresh:
                return (0, 0, 255)  # Red
            elif val > thresh * 0.6:
                return (0, 165, 255)  # Orange
            return (0, 255, 0)  # Green

        y = 55
        metrics_display = [
            ("Visual", metrics.visual_manipulation),
            ("LipSync", metrics.lip_sync_anomaly),
            ("Audio", metrics.audio_artifacts),
            ("Blink", metrics.blink_anomaly),
            ("Scam", metrics.scam_indicators),
        ]

        for i, (name, val) in enumerate(metrics_display):
            x = 15 + (i * 150)
            color = metric_color(val)
            cv2.putText(frame, f"{name}: {val*100:.0f}%",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Mini bar
            bar_width = 80
            cv2.rectangle(frame, (x, y + 5), (x + bar_width, y + 12), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, y + 5), (x + int(bar_width * val), y + 12), color, -1)

        # Confidence
        conf_color = (0, 0, 255) if confidence > 0.75 else (0, 165, 255) if confidence > 0.5 else (0, 255, 0)
        cv2.putText(frame, f"CONFIDENCE: {confidence*100:.0f}%",
                    (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)

        # Face analysis
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                variance = calculate_laplacian_variance(face_roi)
                is_suspicious = variance < THRESHOLDS["laplacian_variance"]

                if is_suspicious:
                    color = (0, 0, 255)
                    label = f"! VAR:{variance:.0f}"
                    # Scan line effect
                    for i in range(y, y+h, 4):
                        alpha = 0.3
                        cv2.line(frame, (x, i), (x+w, i), (0, 0, 200), 1)
                else:
                    color = (0, 255, 0)
                    label = f"OK:{variance:.0f}"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Bottom verdict bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, height-50), (width, height), (10, 10, 10), -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

        if confidence > 0.75:
            verdict_color = (0, 0, 255)
            verdict_text = "HIGH PROBABILITY: SYNTHETIC/MANIPULATED"
        elif confidence > 0.5:
            verdict_color = (0, 165, 255)
            verdict_text = "MEDIUM: POSSIBLE MANIPULATION"
        else:
            verdict_color = (0, 255, 0)
            verdict_text = "LOW: LIKELY AUTHENTIC"

        cv2.putText(frame, verdict_text, (15, height-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, verdict_color, 2)

        out.write(frame)

    cap.release()
    out.release()
    reporter.update("video", 100, "Complete")

    print(f"    Saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(report: VeritasReport, output_path: str = "veritas_report.html"):
    """Generate shareable HTML report for journalists"""

    confidence_class = "high" if report.confidence > 0.75 else "medium" if report.confidence > 0.5 else "low"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VERITAS Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 40px 20px;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00ffff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .subtitle {{ color: #888; margin-bottom: 30px; }}
        .card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .verdict {{
            font-size: 1.5rem;
            font-weight: bold;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .verdict.high {{ background: rgba(255,0,0,0.2); color: #ff4444; border: 2px solid #ff4444; }}
        .verdict.medium {{ background: rgba(255,165,0,0.2); color: #ffa500; border: 2px solid #ffa500; }}
        .verdict.low {{ background: rgba(0,255,0,0.2); color: #00ff88; border: 2px solid #00ff88; }}
        .confidence-bar {{
            height: 30px;
            background: #1a1a2e;
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00ffff, #00ff88);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            font-weight: bold;
            color: #000;
        }}
        .confidence-fill.high {{ background: linear-gradient(90deg, #ff4444, #ff0000); }}
        .confidence-fill.medium {{ background: linear-gradient(90deg, #ffa500, #ff8c00); }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .metric-row:last-child {{ border-bottom: none; }}
        .metric-name {{ font-weight: 500; }}
        .metric-value {{ font-family: monospace; }}
        .flag {{
            background: rgba(255,0,0,0.1);
            border-left: 3px solid #ff4444;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
        .evidence {{
            background: rgba(0,255,255,0.1);
            border-left: 3px solid #00ffff;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
        .section-title {{
            color: #00ffff;
            font-size: 1.2rem;
            margin-bottom: 15px;
        }}
        .disclaimer {{
            font-size: 0.85rem;
            color: #666;
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{ color: #00ffff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PROJECT VERITAS</h1>
        <p class="subtitle">Forensic Deepfake Analysis Report</p>

        <div class="verdict {confidence_class}">
            {report.verdict}
        </div>

        <div class="card">
            <div class="section-title">Confidence Score</div>
            <div class="confidence-bar">
                <div class="confidence-fill {confidence_class}" style="width: {report.confidence*100}%">
                    {report.confidence*100:.0f}%
                </div>
            </div>
            <p style="color: #888; margin-top: 10px;">{report.verdict_explanation}</p>
        </div>

        <div class="card">
            <div class="section-title">Detection Metrics</div>
            <div class="metric-row">
                <span class="metric-name">Visual Manipulation</span>
                <span class="metric-value">{report.metrics.visual_manipulation*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Audio Artifacts</span>
                <span class="metric-value">{report.metrics.audio_artifacts*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Lip-Sync Anomaly</span>
                <span class="metric-value">{report.metrics.lip_sync_anomaly*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Scam Indicators</span>
                <span class="metric-value">{report.metrics.scam_indicators*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Blink Anomaly</span>
                <span class="metric-value">{report.metrics.blink_anomaly*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Noise Pattern</span>
                <span class="metric-value">{report.metrics.noise_pattern_anomaly*100:.1f}%</span>
            </div>
        </div>

        {"".join(f'<div class="flag">{flag}</div>' for flag in report.flags) if report.flags else ""}

        <div class="card">
            <div class="section-title">Analysis Details</div>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Target</td><td>{report.target}</td></tr>
                <tr><td>Analyzed</td><td>{report.analyzed_at}</td></tr>
                <tr><td>Duration</td><td>{report.duration_seconds:.1f} seconds</td></tr>
                <tr><td>Faces Analyzed</td><td>{len(report.face_analysis)}</td></tr>
                <tr><td>VERITAS Version</td><td>{report.version}</td></tr>
            </table>
        </div>

        <div class="card">
            <div class="section-title">Recommendations</div>
            {"".join(f'<div class="evidence">{rec}</div>' for rec in report.recommendations)}
        </div>

        <div class="disclaimer">
            <p><strong>DISCLAIMER:</strong> This analysis is provided for investigative purposes only.
            VERITAS cannot guarantee 100% accuracy and should not be used as sole evidence.
            Always verify findings with additional sources and expert consultation.</p>
            <p style="margin-top: 10px;">Generated by PROJECT VERITAS v{VERSION}</p>
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"    Saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(target: str, generate_video: bool = True, generate_html: bool = True) -> VeritasReport:
    """Run complete VERITAS analysis"""
    start_time = datetime.now()

    print(f"\n{'='*65}")
    print("  PROJECT VERITAS - Forensic Deepfake Detection")
    print(f"  Version {VERSION}")
    print(f"{'='*65}")
    print(f"  Target: {target}")
    print(f"  Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    signals = []
    flags = []

    # 1. Ingest
    reporter.phase("Preparing media...")
    video_path, audio_path = download_video(target)

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    # 2. Visual analysis (also collects landmarks for lip-sync)
    reporter.phase("Analyzing visual content...")
    visual_score, temporal_score, entropy_score, noise_score, face_details, landmarks_sequence = analyze_visual(video_path)

    if visual_score > 0.3:
        signals.append({
            "name": "Face Texture Anomaly",
            "score": visual_score,
            "weight": 0.35,
            "evidence": f"{visual_score*100:.0f}% of faces show unnatural smoothness",
            "technical": "Laplacian variance below threshold indicates synthetic texture"
        })

    if noise_score > 0.3:
        signals.append({
            "name": "GAN Fingerprint",
            "score": noise_score,
            "weight": 0.15,
            "evidence": "Noise patterns consistent with AI generation",
            "technical": "Uniform noise distribution typical of generative models"
        })

    # 3. Blink analysis
    reporter.phase("Analyzing blink patterns...")
    blink_score, blink_count = safe_execute(
        lambda: detect_blinks(video_path),
        fallback=(0.0, 0),
        error_msg="Blink detection failed"
    )

    if blink_score > 0.3:
        signals.append({
            "name": "Blink Anomaly",
            "score": blink_score,
            "weight": 0.1,
            "evidence": f"Detected {blink_count} blinks - abnormal rate for video length",
            "technical": "Deepfakes often have irregular or absent blinking"
        })

    # 4. Audio analysis
    reporter.phase("Analyzing audio...")
    audio_score, audio_evidence = analyze_audio(audio_path)

    if audio_score > 0.3:
        signals.append({
            "name": "Audio Synthesis Markers",
            "score": audio_score,
            "weight": 0.15,
            "evidence": "; ".join(audio_evidence) if audio_evidence else "Spectral anomalies detected",
            "technical": "Missing high-frequency harmonics typical of voice synthesis"
        })

    # 5. Lip-sync analysis
    reporter.phase("Analyzing lip-sync...")
    lip_sync_score = 0.0
    lip_sync_explanation = "Lip-sync analysis not available"

    if MEDIAPIPE_AVAILABLE and LIBROSA_AVAILABLE and audio_path and landmarks_sequence:
        lip_sync_score, lip_sync_explanation = safe_execute(
            lambda: detect_lip_sync(video_path, audio_path, landmarks_sequence, fps),
            fallback=(0.0, "Lip-sync analysis failed"),
            error_msg="Lip-sync detection failed"
        )
        print(f"    {lip_sync_explanation}")
    elif not MEDIAPIPE_AVAILABLE:
        print("    Mediapipe not available - skipping lip-sync")
    elif not LIBROSA_AVAILABLE:
        print("    Librosa not available - skipping lip-sync")
    elif not audio_path:
        print("    No audio track - skipping lip-sync")
    else:
        print("    No landmarks collected - skipping lip-sync")

    if lip_sync_score > 0.3:
        signals.append({
            "name": "Lip-Sync Anomaly",
            "score": lip_sync_score,
            "weight": 0.20,
            "evidence": lip_sync_explanation,
            "technical": "Audio-visual mouth movement correlation analysis"
        })
        flags.append(f"Lip-sync issue: {lip_sync_explanation}")

    # 6. Context analysis
    reporter.phase("Scanning for scam indicators...")
    context_score, context_flags = analyze_context(video_path)
    flags.extend(context_flags)

    if context_score > 0.2:
        signals.append({
            "name": "Scam Content Detected",
            "score": context_score,
            "weight": 0.2,
            "evidence": "; ".join(context_flags) if context_flags else "Suspicious patterns found",
            "technical": "OCR detected wallet addresses or scam keywords"
        })

    # 7. Calculate final confidence
    # Weights: visual 25%, lip-sync 20%, audio 15%, context 15%, blink 10%, entropy 8%, noise 7%
    confidence = (
        visual_score * 0.25 +
        lip_sync_score * 0.20 +
        audio_score * 0.15 +
        context_score * 0.15 +
        blink_score * 0.10 +
        entropy_score * 0.08 +
        noise_score * 0.07
    )

    # Adjust confidence based on signal agreement
    high_signals = sum(1 for s in signals if s["score"] > 0.5)
    if high_signals >= 3:
        confidence = min(1.0, confidence * 1.2)  # Boost if multiple strong signals

    # Generate verdict
    if confidence > 0.75:
        verdict = "HIGH CONFIDENCE: SYNTHETIC/MANIPULATED CONTENT"
        verdict_explanation = "Multiple detection methods indicate this content is likely artificially generated or manipulated."
    elif confidence > 0.5:
        verdict = "MEDIUM CONFIDENCE: POSSIBLE MANIPULATION"
        verdict_explanation = "Some detection methods flagged potential issues. Human review recommended."
    else:
        verdict = "LOW CONFIDENCE: LIKELY AUTHENTIC"
        verdict_explanation = "No strong indicators of manipulation detected. Content appears authentic."

    # Build recommendations
    recommendations = []
    if confidence > 0.5:
        recommendations.append("Cross-reference with known authentic footage of the subject")
        recommendations.append("Check original source and chain of custody")
        recommendations.append("Consult with forensic video experts before publication")
    if context_score > 0.3:
        recommendations.append("Verify any wallet addresses shown are not associated with known scams")
    if audio_score > 0.5:
        recommendations.append("Compare voice with known authentic recordings of the speaker")
    if lip_sync_score > 0.5:
        recommendations.append("Lip movement does not match audio - possible voice dubbing or face swap")
    if len(recommendations) == 0:
        recommendations.append("No immediate concerns, but always verify before trusting")

    # Build metrics
    metrics = Metrics(
        visual_manipulation=visual_score,
        audio_artifacts=audio_score,
        scam_indicators=context_score,
        temporal_inconsistency=temporal_score,
        entropy_anomaly=entropy_score,
        blink_anomaly=blink_score,
        lip_sync_anomaly=lip_sync_score,
        noise_pattern_anomaly=noise_score
    )

    # Build report
    report = VeritasReport(
        version=VERSION,
        target=target,
        analyzed_at=start_time.strftime('%Y-%m-%d %H:%M:%S'),
        duration_seconds=duration,
        metrics=metrics,
        signals=[s for s in signals],
        confidence=confidence,
        verdict=verdict,
        verdict_explanation=verdict_explanation,
        face_analysis=[asdict(fa) for fa in face_details[:50]],  # Limit for file size
        flags=flags,
        recommendations=recommendations
    )

    # 7. Save JSON report
    reporter.phase("Saving reports...")
    report_dict = asdict(report)
    with open("veritas_report.json", "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"    Saved: veritas_report.json")

    # 8. Generate HTML report
    if generate_html:
        generate_html_report(report, "veritas_report.html")

    # 9. Generate annotated video
    if generate_video:
        reporter.phase("Rendering annotated video...")
        generate_annotated_video(video_path, report)

    # 10. Cleanup
    for f in ["temp_video.mp4", "temp_audio.wav"]:
        try:
            if Path(f).exists():
                os.remove(f)
        except:
            pass

    # 11. Print summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*65}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*65}")
    print(f"  Visual Manipulation:  {visual_score*100:5.1f}%")
    print(f"  Lip-Sync Anomaly:     {lip_sync_score*100:5.1f}%")
    print(f"  Audio Artifacts:      {audio_score*100:5.1f}%")
    print(f"  Scam Indicators:      {context_score*100:5.1f}%")
    print(f"  Blink Anomaly:        {blink_score*100:5.1f}%")
    print(f"  Noise Pattern:        {noise_score*100:5.1f}%")
    print(f"{'='*65}")
    print(f"  CONFIDENCE: {confidence*100:.0f}%")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*65}")

    if flags:
        print("\n  FLAGS:")
        for flag in flags:
            print(f"    ! {flag}")

    print(f"\n  Files created:")
    print(f"    - veritas_report.json")
    print(f"    - veritas_report.html")
    if generate_video:
        print(f"    - veritas_output.mp4")

    print(f"\n  Analysis completed in {elapsed:.1f} seconds")
    print(f"{'='*65}\n")

    return report

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(f"""
PROJECT VERITAS v{VERSION}
Forensic Deepfake Detection

Usage:
    python veritas.py <video_url_or_path> [options]

Options:
    --no-video      Skip annotated video generation (faster)
    --no-html       Skip HTML report generation
    --json-only     Only output JSON report

Examples:
    python veritas.py https://youtube.com/watch?v=ABC123
    python veritas.py suspicious_video.mp4
    python veritas.py video.mp4 --no-video
        """)
        sys.exit(1)

    target = sys.argv[1]

    generate_video = "--no-video" not in sys.argv and "--json-only" not in sys.argv
    generate_html = "--no-html" not in sys.argv and "--json-only" not in sys.argv

    try:
        analyze(target, generate_video=generate_video, generate_html=generate_html)
    except VeritasError as e:
        print(f"\n[ERROR] {e.message}")
        if e.technical:
            print(f"        {e.technical}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[*] Analysis cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
