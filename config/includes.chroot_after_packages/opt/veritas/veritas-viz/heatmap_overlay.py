#!/usr/bin/env python3
"""
VERITAS Visualization Engine
Generates annotated video with deepfake detection overlays
"""

import cv2
import json
import numpy as np
import sys
from pathlib import Path

def load_report(report_path="veritas_report.json"):
    """Load the analysis report"""
    with open(report_path, 'r') as f:
        return json.load(f)

def is_abnormally_smooth(face_roi):
    """Check if face region is suspiciously smooth"""
    if face_roi.size == 0:
        return False, 0
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < 120, variance

def process_video(input_path="temp_video.mp4", output_path="veritas_output.mp4"):
    """Generate annotated output video"""

    # Load report
    try:
        report = load_report()
    except FileNotFoundError:
        print("[ERROR] veritas_report.json not found. Run veritas-core first.")
        return

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load face detector
    cascade_paths = [
        "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ]

    face_cascade = None
    for path in cascade_paths:
        if Path(path).exists():
            face_cascade = cv2.CascadeClassifier(path)
            break

    if face_cascade is None:
        print("[WARNING] Face cascade not found. Proceeding without face detection.")

    # Extract metrics
    metrics = report['metrics']
    confidence = report['confidence']
    verdict = report['verdict']

    print(f"[VERITAS-VIZ] Processing {total_frames} frames...")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # === HUD OVERLAY ===

        # Top banner
        cv2.rectangle(frame, (0, 0), (width, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, 220), (0, 255, 255), 2)

        # Title
        cv2.putText(frame, "PROJECT VERITAS // FORENSIC ANALYSIS",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Metrics
        visual_color = (0, 0, 255) if metrics['visual_manipulation'] > 0.5 else (0, 255, 0)
        audio_color = (0, 0, 255) if metrics['audio_artifacts'] > 0.5 else (0, 255, 0)
        scam_color = (0, 0, 255) if metrics['scam_indicators'] > 0.3 else (0, 255, 0)

        cv2.putText(frame, f"Visual Manipulation: {metrics['visual_manipulation']*100:.1f}%",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, visual_color, 2)
        cv2.putText(frame, f"Audio Artifacts: {metrics['audio_artifacts']*100:.1f}%",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, audio_color, 2)
        cv2.putText(frame, f"Scam Indicators: {metrics['scam_indicators']*100:.1f}%",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scam_color, 2)
        cv2.putText(frame, f"Overall Confidence: {confidence*100:.1f}%",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                    (width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # === FACE DETECTION AND ANALYSIS ===

        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                smooth, variance = is_abnormally_smooth(face_roi)

                if smooth:
                    color = (0, 0, 255)  # Red = suspicious
                    label = f"SUSPICIOUS (Var: {variance:.0f})"

                    # Add scan lines for suspicious faces
                    for i in range(y, y+h, 8):
                        cv2.line(frame, (x, i), (x+w, i), (0, 0, 255), 1)
                else:
                    color = (0, 255, 0)  # Green = normal
                    label = f"NORMAL (Var: {variance:.0f})"

                # Draw box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # === BOTTOM VERDICT ===

        cv2.rectangle(frame, (0, height-70), (width, height), (0, 0, 0), -1)

        if confidence > 0.75:
            verdict_color = (0, 0, 255)
            verdict_text = "VERDICT: HIGH PROBABILITY DEEPFAKE / SCAM"
        elif confidence > 0.5:
            verdict_color = (0, 165, 255)
            verdict_text = "VERDICT: POSSIBLE MANIPULATION DETECTED"
        else:
            verdict_color = (0, 255, 0)
            verdict_text = "VERDICT: LIKELY AUTHENTIC"

        cv2.putText(frame, verdict_text, (20, height-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, verdict_color, 3)

        out.write(frame)

        if frame_num % 100 == 0:
            print(f"    Processed {frame_num}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"\n[VERITAS-VIZ] Output saved to: {output_path}")
    print(f"[VERITAS-VIZ] Verdict: {verdict}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "temp_video.mp4"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "veritas_output.mp4"
    process_video(input_file, output_file)
