"""Image Quality Gate and Vegetation Masking."""
import io
from typing import List, Tuple
from PIL import Image
import numpy as np
from app.schemas.quality import ImageQualityResult, AggregateQuality
from app.core.config import settings


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance blur estimator."""
    # Approximate 3x3 Laplacian filter kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0

    # Fast convolution
    pad = np.pad(gray, 1, mode='edge')
    lap = (
        pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * pad[1:-1, 1:-1]
    )
    return float(np.var(lap))


def compute_green_mask_and_coverage(rgb: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Segment green leaf vegetation in HSV color space."""
    # Normalize RGB to [0, 1]
    rgb_norm = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Saturation
    s = np.where(max_c > 0, delta / (max_c + 1e-8), 0.0)
    v = max_c

    # Hue in degrees [0, 360]
    h = np.zeros_like(r)
    mask_r = (max_c == r) & (delta > 0)
    mask_g = (max_c == g) & (delta > 0)
    mask_b = (max_c == b) & (delta > 0)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8)) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8)) + 4
    h = h * 60.0

    # Green vegetation bounds (H: 35-85, S > 0.20, V > 0.20)
    green_mask = (h >= 35.0) & (h <= 85.0) & (s >= 0.20) & (v >= 0.20)
    total_pixels = rgb.shape[0] * rgb.shape[1]
    green_pixels = int(np.sum(green_mask))
    coverage = float(green_pixels / max(1, total_pixels))

    return green_mask, coverage, green_pixels


def compute_perceptual_hash(gray: np.ndarray) -> str:
    """Compute 64-bit average perceptual hash for duplicate detection."""
    img = Image.fromarray(gray).resize((8, 8), Image.Resampling.BILINEAR)
    pixels = np.array(img, dtype=np.float32)
    avg = np.mean(pixels)
    bits = (pixels > avg).flatten()
    return "".join(["1" if b else "0" for b in bits])


def hamming_distance(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


class DefaultQualityGate:
    """Default implementation of Image Quality Gate."""
    def __init__(self, blur_threshold: float = 35.0, min_resolution: int = 224):
        self.blur_threshold = blur_threshold
        self.min_resolution = min_resolution

    def evaluate_image(self, image: Image.Image, index: int) -> ImageQualityResult:
        rgb_img = image.convert("RGB")
        w, h = rgb_img.size
        issues: List[str] = []

        if w < self.min_resolution or h < self.min_resolution:
            issues.append(f"Resolution too low: {w}x{h} (minimum {self.min_resolution}x{self.min_resolution})")

        rgb_arr = np.array(rgb_img)
        gray = np.array(rgb_img.convert("L"))

        # 1. Blur
        blur_score = compute_laplacian_variance(gray)
        is_blurry = blur_score < self.blur_threshold
        if is_blurry:
            issues.append(f"Image is blurry (score: {blur_score:.1f}, threshold: {self.blur_threshold:.1f})")

        # 2. Exposure
        mean_brightness = float(np.mean(gray))
        is_underexposed = mean_brightness < 30.0
        is_overexposed = mean_brightness > 225.0
        if is_underexposed:
            issues.append("Image is severely underexposed/dark")
        if is_overexposed:
            issues.append("Image is severely overexposed/washed out")

        # 3. Leaf coverage
        _, leaf_cov, leaf_mask_area = compute_green_mask_and_coverage(rgb_arr)
        if leaf_cov < 0.25:
            issues.append(f"Low leaf coverage ({leaf_cov*100:.1f}%). Center the leaf in the frame.")

        # Compute per-image quality score (0.0 - 1.0)
        q = 1.0
        if is_blurry:
            q -= 0.35
        if is_underexposed or is_overexposed:
            q -= 0.30
        if leaf_cov < 0.25:
            q -= 0.25
        quality_score = max(0.1, min(1.0, q))

        is_usable = (len(issues) == 0) and (quality_score >= 0.50)

        return ImageQualityResult(
            image_index=index,
            is_usable=is_usable,
            blur_score=blur_score,
            is_blurry=is_blurry,
            mean_brightness=mean_brightness,
            is_underexposed=is_underexposed,
            is_overexposed=is_overexposed,
            is_duplicate=False,
            leaf_coverage=leaf_cov,
            leaf_mask_area=leaf_mask_area,
            quality_score=quality_score,
            issues=issues
        )

    def evaluate_batch(self, images: List[Image.Image]) -> AggregateQuality:
        if len(images) < 3 or len(images) > 5:
            return AggregateQuality(
                overall_quality_score=0.20,
                passes_gate=False,
                reasons=[f"Must provide 3 to 5 images (received {len(images)})."],
                recapture_guidance=["Please upload between 3 and 5 clear photos of the affected plant leaves."]
            )

        results = [self.evaluate_image(img, i) for i, img in enumerate(images)]

        # Check for duplicates using perceptual hashing
        hashes = [compute_perceptual_hash(np.array(img.convert("L"))) for img in images]
        has_dup = False
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                dist = hamming_distance(hashes[i], hashes[j])
                if dist < 5:
                    has_dup = True
                    results[j] = results[j].model_copy(update={"is_duplicate": True})

        scores = [r.quality_score for r in results]
        overall_score = float(np.mean(scores))
        if has_dup:
            overall_score = max(0.2, overall_score - 0.25)

        passes = (overall_score >= settings.QUALITY_MIN) and not has_dup

        reasons = []
        guidance = []
        if not passes:
            reasons.append(f"Aggregate quality score ({overall_score:.2f}) is below minimum threshold ({settings.QUALITY_MIN:.2f}).")
            if has_dup:
                reasons.append("Duplicate or near-identical images detected.")
                guidance.append("Capture multiple distinct leaves and angles rather than duplicates of the same spot.")
            guidance.append("Hold the phone steady to avoid motion blur.")
            guidance.append("Ensure indirect natural lighting without direct harsh glare or deep shadow.")
            guidance.append("Fill at least 50% of the camera view with the leaf surface.")

        return AggregateQuality(
            overall_quality_score=overall_score,
            passes_gate=passes,
            reasons=reasons,
            recapture_guidance=guidance
        )
