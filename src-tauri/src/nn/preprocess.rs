#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_background_handwriting_is_inverted_to_bright_ink() {
        // 26x26 white canvas with a dark vertical stroke: the border is
        // bright, so the stroke must become the bright signal.
        let mut img = vec![255u8; 26 * 26];
        for y in 4..22 {
            for x in 11..15 {
                img[y * 26 + x] = 30;
            }
        }
        let out = preprocess(&img, 26, 26);
        // Background cells must be dark, stroke cells bright.
        assert!(out[0] < 0.1, "corner should be background: {}", out[0]);
        let center_max = out.iter().fold(0.0f64, |a, &v| a.max(v));
        assert!(center_max > 0.5, "stroke ink should survive: {}", center_max);
    }

    #[test]
    fn dark_background_image_is_not_inverted() {
        let mut img = vec![0u8; 26 * 26];
        for y in 4..22 {
            for x in 11..15 {
                img[y * 26 + x] = 200;
            }
        }
        let out = preprocess(&img, 26, 26);
        assert!(out[0] < 0.1);
        let center_max = out.iter().fold(0.0f64, |a, &v| a.max(v));
        assert!(center_max > 0.5);
    }

    #[test]
    fn zero_image_center_of_mass_is_canvas_center() {
        let img = vec![0u8; 26 * 26];
        let (cx, cy) = center_of_mass(&img, 26, 26);
        assert!((cx - 12.5).abs() < 1e-9);
        assert!((cy - 12.5).abs() < 1e-9);
    }

    #[test]
    fn single_bright_pixel_center_of_mass_is_at_pixel() {
        let mut img = vec![0u8; 26 * 26];
        img[5 * 26 + 9] = 255;
        let (cx, cy) = center_of_mass(&img, 26, 26);
        assert!((cx - 9.0).abs() < 1e-9);
        assert!((cy - 5.0).abs() < 1e-9);
    }

    #[test]
    fn resampled_center_of_mass_is_grid_center() {
        let mut img = vec![0u8; 52 * 52];
        // A vertical bar left of center: after centering it should land mid-grid.
        for y in 10..40 {
            img[y * 52 + 10] = 200;
            img[y * 52 + 11] = 200;
        }
    let out = preprocess(&img, 52, 52);
    assert_eq!(out.len(), 256);
    let (cx, cy) = center_of_mass_f64(&out, 16, 16);
    assert!((cx - 7.5).abs() < 0.51, "cx = {}", cx);
    assert!((cy - 7.5).abs() < 0.51, "cy = {}", cy);
    }

    #[test]
    fn output_is_normalized_0_to_1() {
        let img: Vec<u8> = (0..256u32).map(|i| (i % 256) as u8).collect();
        let out = preprocess(&img, 16, 16);
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    /// Locks the cross-language parity contract: the JS engine asserts the
    /// same golden values (generated from this implementation). Changing
    /// preprocess math requires regenerating both sides together.
    #[test]
    fn golden_values_parity_with_js() {
        let mut img = vec![0u8; 25];
        for i in 0..5 {
            img[i * 5 + i] = 255;
            img[i * 5 + 4 - i] = 128;
        }
        let out = preprocess(&img, 5, 5);
        assert_eq!(out.len(), 256);
        // Spot-check values the JS test verifies in full.
        assert!((out[0] - 0.4306640625).abs() < 1e-12);
        assert!((out[1] - 0.6357421875).abs() < 1e-12);
        assert!((out[255] - out[0]).abs() < 1e-12);

        let mut img2 = vec![0u8; 26 * 26];
        for y in 0..26usize {
            for x in 0..26usize {
                let dx = x as f64 - 13.0;
                let dy = y as f64 - 13.0;
                let d = (dx * dx + dy * dy).sqrt();
                if d < 6.0 {
                    img2[y * 26 + x] = (255.0 * (1.0 - d / 6.0)) as u8;
                }
            }
        }
        let out2 = preprocess(&img2, 26, 26);
        assert!((out2[0] - 0.0).abs() < 1e-12);
        assert!(out2.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
/// Intensity-weighted centroid in pixel coordinates. A blank image returns
/// the canvas center so downstream centering is a no-op.
pub fn center_of_mass(img: &[u8], w: usize, h: usize) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let v = img[y * w + x] as f64;
            sum += v;
            sx += x as f64 * v;
            sy += y as f64 * v;
        }
    }
    if sum == 0.0 {
        ((w - 1) as f64 / 2.0, (h - 1) as f64 / 2.0)
    } else {
        (sx / sum, sy / sum)
    }
}

/// Same centroid math over the preprocessed f64 grid (used by tests and UI).
pub const GRID: usize = 16;

pub fn center_of_mass_f64(img: &[f64], w: usize, h: usize) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let v = img[y * w + x];
            sum += v;
            sx += x as f64 * v;
            sy += y as f64 * v;
        }
    }
    if sum == 0.0 {
        ((w - 1) as f64 / 2.0, (h - 1) as f64 / 2.0)
    } else {
        (sx / sum, sy / sum)
    }
}

/// Translate so the centroid lands on the canvas center, then bilinearly
/// resample to a 16x16 grid with values normalized to [0, 1].
pub fn preprocess(img: &[u8], w: usize, h: usize) -> Vec<f64> {
    const OUT: usize = GRID;
    // Handwriting pads are black-ink-on-white while imported MNIST PNGs are
    // white-ink-on-black. Detect polarity from the border ring so both look
    // like bright strokes on a dark background to the network.
    let normalized = normalize_polarity(img, w, h);
    let (cx, cy) = center_of_mass(&normalized, w, h);
    let dx = (w - 1) as f64 / 2.0 - cx;
    let dy = (h - 1) as f64 / 2.0 - cy;
    let mut out = vec![0.0f64; OUT * OUT];
    // Map each output cell center back to source coordinates, then sample
    // the centered image bilinearly.
    let scale_x = w as f64 / OUT as f64;
    let scale_y = h as f64 / OUT as f64;
    for oy in 0..OUT {
        for ox in 0..OUT {
            // Source coordinate of the output cell center, minus the shift.
            let sx = (ox as f64 + 0.5) * scale_x - dx - 0.5;
            let sy = (oy as f64 + 0.5) * scale_y - dy - 0.5;
            let v = bilinear(&normalized, w, h, sx, sy);
            out[oy * OUT + ox] = v / 255.0;
        }
    }
    out
}

/// Flips white-background images to dark-background so ink is always the
/// bright signal. Decides from the mean of a 2px border ring.
fn normalize_polarity(img: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let border = x < 2 || y < 2 || x + 2 >= w || y + 2 >= h;
            if border {
                sum += img[y * w + x] as f64;
                n += 1.0;
            }
        }
    }
    if n > 0.0 && sum / n > 127.0 {
        img.iter().map(|&v| 255 - v).collect()
    } else {
        img.to_vec()
    }
}

/// Bilinear sample with clamped edges; outside the canvas reads as zero.
fn bilinear(img: &[u8], w: usize, h: usize, x: f64, y: f64) -> f64 {
    if x < -0.5 || y < -0.5 || x > w as f64 - 0.5 || y > h as f64 - 0.5 {
        return 0.0;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let x0 = x0 as isize;
    let y0 = y0 as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let at = |xx: isize, yy: isize| -> f64 {
        if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
            0.0
        } else {
            img[yy as usize * w + xx as usize] as f64
        }
    };
    let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let bot = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}
