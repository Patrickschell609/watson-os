use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

mod ingest;
mod vision;
mod audio;
mod context;

#[derive(Serialize, Deserialize)]
pub struct VeritasReport {
    pub target: String,
    pub metrics: Metrics,
    pub confidence: f32,
    pub verdict: String,
}

#[derive(Serialize, Deserialize)]
pub struct Metrics {
    pub visual_manipulation: f32,
    pub audio_artifacts: f32,
    pub scam_indicators: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let target = args.get(1).expect("Usage: veritas-core <video_url_or_path>");

    println!("[VERITAS] Analyzing: {}", target);

    // 1. Ingest
    println!("[*] Downloading/preparing streams...");
    let (video_path, audio_path) = ingest::download_stream(target).await?;

    // 2. Visual Analysis
    println!("[*] Analyzing visual frames...");
    let visual_score = vision::analyze_video(&video_path)?;
    println!("    Visual manipulation score: {:.2}", visual_score);

    // 3. Audio Analysis
    println!("[*] Analyzing audio...");
    let audio_score = audio::analyze_audio(&audio_path)?;
    println!("    Audio artifact score: {:.2}", audio_score);

    // 4. Context Analysis
    println!("[*] Scanning for scam indicators...");
    let context_score = context::scan_video_text(&video_path)?;
    println!("    Scam indicator score: {:.2}", context_score);

    // 5. Calculate confidence
    let confidence = (visual_score * 0.5) + (audio_score * 0.25) + (context_score * 0.25);
    let verdict = if confidence > 0.75 {
        "HIGH CONFIDENCE: SYNTHETIC/MANIPULATED CONTENT"
    } else if confidence > 0.5 {
        "MEDIUM CONFIDENCE: POSSIBLE MANIPULATION"
    } else {
        "LOW CONFIDENCE: LIKELY AUTHENTIC"
    };

    // 6. Generate report
    let report = VeritasReport {
        target: target.clone(),
        metrics: Metrics {
            visual_manipulation: visual_score,
            audio_artifacts: audio_score,
            scam_indicators: context_score,
        },
        confidence,
        verdict: verdict.to_string(),
    };

    let report_json = serde_json::to_string_pretty(&report)?;
    fs::write("veritas_report.json", &report_json)?;
    println!("\n[VERITAS] Report saved to veritas_report.json");
    println!("{}", report_json);

    // 7. Cleanup
    ingest::cleanup()?;

    println!("\n[VERITAS] Analysis complete. Run veritas-viz for visual overlay.");
    Ok(())
}
