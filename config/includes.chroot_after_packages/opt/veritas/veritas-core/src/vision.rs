use anyhow::Result;
use opencv::{
    core::{Mat, Size, Rect, Vector},
    imgproc,
    objdetect::CascadeClassifier,
    videoio::{VideoCapture, CAP_ANY, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES},
    prelude::*,
};

pub fn analyze_video(path: &str) -> Result<f32> {
    let mut cap = VideoCapture::from_file(path, CAP_ANY)?;

    if !cap.is_opened()? {
        return Err(anyhow::anyhow!("Could not open video file"));
    }

    // Load face detector - try multiple paths
    let cascade_paths = [
        "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ];

    let mut face_cascade = CascadeClassifier::default()?;
    for path in &cascade_paths {
        if std::path::Path::new(path).exists() {
            face_cascade = CascadeClassifier::new(path)?;
            if !face_cascade.empty() {
                break;
            }
        }
    }

    let frame_count = cap.get(CAP_PROP_FRAME_COUNT)? as i32;
    let sample_every = 10; // Sample every 10th frame
    let max_frames = 200;

    let mut suspicious_faces = 0;
    let mut total_faces = 0;
    let mut frame = Mat::default();
    let mut frame_idx = 0;

    println!("    Scanning {} frames (sampling every {})...", frame_count, sample_every);

    while cap.read(&mut frame)? {
        frame_idx += 1;

        if frame_idx % sample_every != 0 {
            continue;
        }

        if frame.empty() {
            continue;
        }

        // Convert to grayscale for face detection
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Detect faces
        let mut faces = Vector::<Rect>::new();
        face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            3,
            0,
            Size::new(30, 30),
            Size::new(0, 0),
        )?;

        for face in faces.iter() {
            total_faces += 1;

            // Extract face ROI
            let roi = Mat::roi(&gray, face)?;

            // Heuristic 1: Check for abnormal smoothness (low texture)
            if is_abnormally_smooth(&roi)? {
                suspicious_faces += 1;
                continue;
            }

            // Heuristic 2: Check for boundary color inconsistency
            if has_boundary_inconsistency(&frame, face)? {
                suspicious_faces += 1;
            }
        }

        if frame_idx / sample_every > max_frames {
            break;
        }
    }

    if total_faces == 0 {
        println!("    Warning: No faces detected in video");
        return Ok(0.0);
    }

    let score = suspicious_faces as f32 / total_faces as f32;
    println!("    Analyzed {} faces, {} suspicious ({:.1}%)",
             total_faces, suspicious_faces, score * 100.0);

    Ok(score)
}

fn is_abnormally_smooth(roi: &Mat) -> Result<bool> {
    // Calculate Laplacian variance (measure of texture/sharpness)
    let mut laplacian = Mat::default();
    imgproc::laplacian(&roi, &mut laplacian, opencv::core::CV_64F, 1, 1.0, 0.0, opencv::core::BORDER_DEFAULT)?;

    let mut mean = opencv::core::Scalar::default();
    let mut stddev = opencv::core::Scalar::default();
    opencv::core::mean_std_dev(&laplacian, &mut mean, &mut stddev, &Mat::default())?;

    let variance = stddev[0] * stddev[0];

    // Deepfakes often have unnaturally smooth faces
    // Real faces typically have variance > 100-150
    Ok(variance < 120.0)
}

fn has_boundary_inconsistency(frame: &Mat, face: Rect) -> Result<bool> {
    let margin = 15;
    let frame_cols = frame.cols();
    let frame_rows = frame.rows();

    // Expand ROI for boundary check
    let expanded = Rect::new(
        (face.x - margin).max(0),
        (face.y - margin).max(0),
        (face.width + 2 * margin).min(frame_cols - face.x.max(0)),
        (face.height + 2 * margin).min(frame_rows - face.y.max(0)),
    );

    // Get mean colors
    let face_roi = Mat::roi(frame, face)?;
    let boundary_roi = Mat::roi(frame, expanded)?;

    let face_mean = opencv::core::mean(&face_roi, &Mat::default())?;
    let boundary_mean = opencv::core::mean(&boundary_roi, &Mat::default())?;

    // Calculate color distance
    let mut diff = 0.0;
    for i in 0..3 {
        let d = face_mean[i] - boundary_mean[i];
        diff += d * d;
    }
    let distance = diff.sqrt();

    // Large color difference at boundary = suspicious
    Ok(distance > 40.0)
}
