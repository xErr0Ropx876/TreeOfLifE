"""File Storage and EXIF Stripping."""
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple
from PIL import Image
from app.core.config import settings


class StorageService:
    """Manages uploaded images, crops, EXIF stripping, and retention timestamps."""
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.crops_dir = Path(settings.CROPS_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def save_and_strip_exif(self, image_bytes: bytes, filename: str) -> Tuple[Path, str, datetime]:
        """Verify image, strip EXIF metadata (GPS, camera info), save cleanly."""
        file_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"

        dest_path = self.upload_dir / f"{file_id}{ext}"

        # Pillow open and re-save without EXIF
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(list(img.getdata()))

        # Save stripped image
        clean_img.save(dest_path, quality=95)

        retention_expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.RETENTION_HOURS)
        return dest_path, file_id, retention_expires_at


storage_service = StorageService()
