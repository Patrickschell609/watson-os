use anyhow::Result;
use regex::Regex;
use opencv::{
    core::Mat,
    videoio::{VideoCapture, CAP_ANY, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES},
    imgcodecs,
    prelude::*,
};
use std::process::Command;
use std::fs;

pub fn scan_video_text(video_path: &str) -> Result<f32> {
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    let frame_count = cap.get(CAP_PROP_FRAME_COUNT)? as i64;
    let num_samples = 12;

    let scam_keywords = [
        "giveaway", "double", "2x", "return", "send eth", "send btc",
        "urgent", "limited", "elon", "tesla", "free", "claim",
        "airdrop", "winner", "congratulations", "act now"
    ];

    let wallet_regex = Regex::new(r"0x[a-fA-F0-9]{40}")?;
    let btc_regex = Regex::new(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}")?;

    let mut full_text = String::new();
    let mut frame = Mat::default();

    println!("    Extracting text from {} sample frames...", num_samples);

    for i in 0..num_samples {
        let pos = (i as f64 * frame_count as f64 / num_samples as f64) as f64;
        cap.set(CAP_PROP_POS_FRAMES, pos)?;

        if !cap.read(&mut frame)? || frame.empty() {
            continue;
        }

        let temp_path = format!("temp_ocr_{}.png", i);
        imgcodecs::imwrite(&temp_path, &frame, &opencv::core::Vector::new())?;

        // Run tesseract OCR
        let output = Command::new("tesseract")
            .args(&[&temp_path, "stdout", "-l", "eng", "--psm", "3"])
            .output();

        if let Ok(result) = output {
            let text = String::from_utf8_lossy(&result.stdout).to_lowercase();
            full_text.push_str(&text);
            full_text.push(' ');
        }

        let _ = fs::remove_file(&temp_path);
    }

    // Calculate scam score
    let mut score = 0.0;

    // Check for crypto wallet addresses
    if wallet_regex.is_match(&full_text) {
        println!("    Found ETH wallet address!");
        score += 0.5;
    }

    if btc_regex.is_match(&full_text) {
        println!("    Found BTC wallet address!");
        score += 0.5;
    }

    // Check for scam keywords
    let mut keywords_found = Vec::new();
    for keyword in &scam_keywords {
        if full_text.contains(keyword) {
            keywords_found.push(*keyword);
            score += 0.1;
        }
    }

    if !keywords_found.is_empty() {
        println!("    Scam keywords found: {:?}", keywords_found);
    }

    Ok(score.min(1.0))
}
