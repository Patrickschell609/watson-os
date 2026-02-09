use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use rustfft::{FftPlanner, num_complex::Complex};

pub fn analyze_audio(path: &str) -> Result<f32> {
    // Read WAV file
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Parse WAV header manually (simple implementation)
    let samples = read_wav_samples(reader)?;

    if samples.is_empty() {
        println!("    Warning: No audio samples found");
        return Ok(0.0);
    }

    let sample_rate = 44100;
    let fft_size = 2048;
    let high_cutoff_hz = 12000;
    let bin_cutoff = high_cutoff_hz * fft_size / sample_rate;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut anomalies = 0.0;
    let mut chunks = 0.0;

    for chunk in samples.chunks(fft_size) {
        if chunk.len() < fft_size {
            break;
        }

        let mut buffer: Vec<Complex<f32>> = chunk
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Calculate high-frequency energy ratio
        let mut high_energy = 0.0;
        let mut total_energy = 0.0;

        for (i, c) in buffer.iter().enumerate().take(fft_size / 2) {
            let mag = c.norm();
            let energy = mag * mag;
            total_energy += energy;

            if i > bin_cutoff {
                high_energy += energy;
            }
        }

        let high_ratio = if total_energy > 0.0 {
            high_energy / total_energy
        } else {
            0.0
        };

        // Very low high-frequency content = suspicious (voice synthesis artifact)
        if high_ratio < 0.05 {
            anomalies += 1.0;
        }

        chunks += 1.0;

        if chunks > 200.0 {
            break;
        }
    }

    if chunks == 0.0 {
        return Ok(0.0);
    }

    let score = anomalies / chunks;
    println!("    Analyzed {} audio chunks, {:.0} anomalous ({:.1}%)",
             chunks as i32, anomalies, score * 100.0);

    Ok(score)
}

fn read_wav_samples(mut reader: BufReader<File>) -> Result<Vec<f32>> {
    use std::io::Read;

    // Skip WAV header (44 bytes for standard WAV)
    let mut header = [0u8; 44];
    reader.read_exact(&mut header)?;

    // Read samples as i16 and convert to f32
    let mut samples = Vec::new();
    let mut buffer = [0u8; 2];

    while reader.read_exact(&mut buffer).is_ok() {
        let sample = i16::from_le_bytes(buffer);
        samples.push(sample as f32 / 32768.0);

        // Limit samples for performance
        if samples.len() > 44100 * 60 {
            break;
        }
    }

    Ok(samples)
}
