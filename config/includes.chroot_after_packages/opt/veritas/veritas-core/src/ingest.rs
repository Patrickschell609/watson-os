use std::process::Command;
use std::path::Path;
use anyhow::{Context, Result};
use std::fs;

pub async fn download_stream(target: &str) -> Result<(String, String)> {
    let video_out = "temp_video.mp4".to_string();
    let audio_out = "temp_audio.wav".to_string();

    let is_url = target.starts_with("http://") || target.starts_with("https://") || target.starts_with("www.");

    if is_url {
        println!("    Downloading from URL...");

        // Download video
        let video_result = Command::new("yt-dlp")
            .args(&["-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", "-o", &video_out, target])
            .output()
            .context("Failed to run yt-dlp. Install with: pip install yt-dlp")?;

        if !video_result.status.success() {
            println!("    Warning: yt-dlp failed, trying direct download...");
            Command::new("curl")
                .args(&["-L", "-o", &video_out, target])
                .output()?;
        }

        // Extract audio
        Command::new("ffmpeg")
            .args(&["-i", &video_out, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", &audio_out, "-y"])
            .output()
            .context("Failed to extract audio with ffmpeg")?;

    } else {
        println!("    Processing local file...");

        if !Path::new(target).exists() {
            return Err(anyhow::anyhow!("Local file does not exist: {}", target));
        }

        fs::copy(target, &video_out)?;

        Command::new("ffmpeg")
            .args(&["-i", target, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", &audio_out, "-y"])
            .output()
            .context("Failed to extract audio with ffmpeg")?;
    }

    Ok((video_out, audio_out))
}

pub fn cleanup() -> Result<()> {
    let _ = fs::remove_file("temp_video.mp4");
    let _ = fs::remove_file("temp_audio.wav");

    // Clean up any temp OCR frames
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("temp_ocr_") && name.ends_with(".png") {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    Ok(())
}
