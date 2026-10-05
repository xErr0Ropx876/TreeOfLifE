"""DPDP Act Automated Retention Janitor."""
from datetime import datetime, timezone
from pathlib import Path
from app.db.mongo import db_manager


class JanitorService:
    """Purges expired files and database records, preserving open expert escalations."""
    async def cleanup_expired_records(self) -> int:
        if db_manager.async_db is None:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        db = db_manager.async_db

        # 1. Find all open escalations (these are exempt from deletion)
        open_escalations = await db.expert_escalations.find({"status": "open"}).to_list(1000)
        exempt_request_ids = {e.get("request_id") for e in open_escalations if e.get("request_id")}

        # 2. Find expired requests
        expired_cursor = db.prediction_requests.find({
            "retention_expires_at": {"$lt": now}
        })
        expired_requests = await expired_cursor.to_list(1000)

        deleted_count = 0
        for req in expired_requests:
            req_id = req.get("request_id")
            if req_id in exempt_request_ids:
                # Open escalation exists; keep records
                continue

            # Remove image files from disk
            images = await db.uploaded_images.find({"request_id": req_id}).to_list(10)
            for img in images:
                file_path = img.get("file_path")
                if file_path:
                    p = Path(file_path)
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

            # Cascade delete in MongoDB
            await db.uploaded_images.delete_many({"request_id": req_id})
            await db.image_quality_results.delete_many({"request_id": req_id})
            await db.florence_detections.delete_many({"request_id": req_id})
            await db.qwen_phenotypes.delete_many({"request_id": req_id})
            await db.efficientnet_predictions.delete_many({"request_id": req_id})
            await db.fusion_results.delete_many({"request_id": req_id})
            await db.prediction_requests.delete_one({"request_id": req_id})
            deleted_count += 1

        return deleted_count


janitor = JanitorService()
