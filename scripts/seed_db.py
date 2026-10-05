"""Idempotent seed script to populate MongoDB with Karnataka rice varieties and genomic priors."""
from datetime import datetime, timezone
from pymongo import MongoClient, ReplaceOne
from app.core.config import settings, load_yaml


def seed_mongodb():
    print(f"Connecting to MongoDB at {settings.MONGO_URI}...")
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]

    varieties_yaml = load_yaml("configs/varieties_seed.yaml")
    varieties = varieties_yaml.get("varieties", [])

    if not varieties:
        print("No varieties found in configs/varieties_seed.yaml.")
        return

    variety_ops = []
    genomic_ops = []

    now = datetime.now(timezone.utc).isoformat()

    for v in varieties:
        cid = v["canonical_id"]

        # Variety Document
        var_doc = {
            "canonical_id": cid,
            "canonical_name": v["canonical_name"],
            "aliases": v.get("aliases", []),
            "region": v.get("region", "Karnataka"),
            "duration_days": v.get("duration_days", 120),
            "updated_at": now
        }
        variety_ops.append(ReplaceOne({"canonical_id": cid}, var_doc, upsert=True))

        # Genomic Prior Document
        gen_doc = {
            "canonical_variety_id": cid,
            "canonical_name": v["canonical_name"],
            "is_placeholder": v.get("is_placeholder", True),
            "source_database": v.get("source_database", "local_curated"),
            "source_version": v.get("source_version", "2024.1"),
            "disclaimer": "This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant.",
            "susceptibility": v.get("susceptibility", {}),
            "updated_at": now
        }
        genomic_ops.append(ReplaceOne({"canonical_variety_id": cid}, gen_doc, upsert=True))

    if variety_ops:
        db.varieties.bulk_write(variety_ops)
        print(f"Successfully seeded {len(variety_ops)} varieties into MongoDB collection 'varieties'.")

    if genomic_ops:
        db.genomic_profiles.bulk_write(genomic_ops)
        print(f"Successfully seeded {len(genomic_ops)} genomic profiles into MongoDB collection 'genomic_profiles'.")

    client.close()
    print("Database seeding completed.")


if __name__ == "__main__":
    seed_mongodb()
