"""Florence-2-large Lesion Localization and OCR Adapter."""
import gc
from typing import List, Optional
from PIL import Image
from app.schemas.localizer import LocalizerResult, BoundingBox
from app.services.localizer.classical import ClassicalLocalizer
from app.pipeline.gpu_lock import gpu_lock


class FlorenceLocalizer:
    """Real Florence-2 adapter for lesion localization and seed packet OCR."""
    def __init__(self, model_id: str = "microsoft/Florence-2-large"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.classical_fallback = ClassicalLocalizer()

    async def load(self) -> None:
        await gpu_lock.acquire("florence-2")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(device)
            self.model.eval()
        except Exception as e:
            print(f"[FlorenceLocalizer] Failed to load {self.model_id}: {e}. Falling back to Classical Localizer.")
            self.model = None
            self.processor = None

    async def unload(self) -> None:
        try:
            if self.model is not None:
                del self.model
                del self.processor
                self.model = None
                self.processor = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            gpu_lock.release()

    async def localize_and_ocr(
        self,
        leaf_images: List[Image.Image],
        seed_packet_image: Optional[Image.Image] = None
    ) -> List[LocalizerResult]:
        if self.model is None or self.processor is None:
            return await self.classical_fallback.localize_and_ocr(leaf_images, seed_packet_image)

        try:
            import torch
            device = next(self.model.parameters()).device
            results: List[LocalizerResult] = []

            # 1. Detect lesions
            task_prompt = "<OD>"
            for idx, img in enumerate(leaf_images):
                w_img, h_img = img.size
                inputs = self.processor(text=task_prompt, images=img, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512,
                        num_beams=3
                    )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = self.processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(w_img, h_img)
                )

                boxes = parsed.get("<OD>", {}).get("bboxes", [])
                labels = parsed.get("<OD>", {}).get("labels", [])

                detections: List[BoundingBox] = []
                for box, lbl in zip(boxes[:5], labels[:5]):
                    xmin, ymin, xmax, ymax = box
                    detections.append(BoundingBox(
                        xmin=int(xmin),
                        ymin=int(ymin),
                        xmax=int(xmax),
                        ymax=int(ymax),
                        confidence=0.85,
                        label=str(lbl)
                    ))

                # OCR on seed packet in same cycle if present
                ocr_text = None
                if seed_packet_image is not None and idx == 0:
                    ocr_inputs = self.processor(text="<OCR_WITH_REGION>", images=seed_packet_image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        ocr_ids = self.model.generate(
                            input_ids=ocr_inputs["input_ids"],
                            pixel_values=ocr_inputs["pixel_values"],
                            max_new_tokens=256
                        )
                    ocr_res = self.processor.batch_decode(ocr_ids, skip_special_tokens=True)[0]
                    ocr_text = ocr_res.replace("<OCR_WITH_REGION>", "").strip()

                results.append(LocalizerResult(
                    image_index=idx,
                    detections=detections,
                    lesion_count=len(detections),
                    crops_available=len(detections) > 0,
                    ocr_text=ocr_text,
                    lesion_localization_unavailable=len(detections) == 0
                ))

            return results
        except Exception as e:
            print(f"[FlorenceLocalizer] Inference error: {e}. Falling back to Classical Localizer.")
            return await self.classical_fallback.localize_and_ocr(leaf_images, seed_packet_image)
