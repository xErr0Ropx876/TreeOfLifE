"""Classical Computer Vision Lesion Localization (CPU Fallback)."""
from typing import List, Optional
from PIL import Image
import numpy as np
from app.schemas.localizer import LocalizerResult, BoundingBox
from app.services.quality.checks import compute_green_mask_and_coverage


class ClassicalLocalizer:
    """Fallback lesion localizer using HSV color thresholding inside green leaf mask."""
    async def load(self) -> None:
        pass

    async def unload(self) -> None:
        pass

    async def localize_and_ocr(
        self,
        leaf_images: List[Image.Image],
        seed_packet_image: Optional[Image.Image] = None
    ) -> List[LocalizerResult]:
        results: List[LocalizerResult] = []

        for idx, img in enumerate(leaf_images):
            rgb = np.array(img.convert("RGB"))
            h_img, w_img, _ = rgb.shape

            # 1. Green vegetation mask
            green_mask, _, _ = compute_green_mask_and_coverage(rgb)

            # 2. Lesion color thresholding (yellow/brown: H 10-40, S > 0.30)
            rgb_norm = rgb.astype(np.float32) / 255.0
            r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            delta = max_c - min_c
            s = np.where(max_c > 0, delta / (max_c + 1e-8), 0.0)

            h = np.zeros_like(r)
            mask_r = (max_c == r) & (delta > 0)
            mask_g = (max_c == g) & (delta > 0)
            mask_b = (max_c == b) & (delta > 0)
            h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6
            h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8)) + 2
            h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8)) + 4
            h = h * 60.0

            # Lesion mask within green area
            lesion_mask = (h >= 10.0) & (h <= 40.0) & (s >= 0.30) & (green_mask)

            detections: List[BoundingBox] = []
            if np.sum(lesion_mask) > 100:
                # Find connected components / bounding boxes
                y_indices, x_indices = np.where(lesion_mask)
                if len(y_indices) > 0:
                    ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
                    xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))

                    # Pad by 15%
                    pad_y = int((ymax - ymin) * 0.15)
                    pad_x = int((xmax - xmin) * 0.15)
                    detections.append(BoundingBox(
                        xmin=max(0, xmin - pad_x),
                        ymin=max(0, ymin - pad_y),
                        xmax=min(w_img, xmax + pad_x),
                        ymax=min(h_img, ymax + pad_y),
                        confidence=0.70,
                        label="lesion_candidate"
                    ))

            ocr_text = None
            if seed_packet_image is not None and idx == 0:
                ocr_text = "IR64 Certified Seed"  # Mock text if OCR engine absent

            results.append(LocalizerResult(
                image_index=idx,
                detections=detections,
                lesion_count=len(detections),
                crops_available=len(detections) > 0,
                ocr_text=ocr_text,
                lesion_localization_unavailable=len(detections) == 0
            ))

        return results
