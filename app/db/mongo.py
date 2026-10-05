"""MongoDB database client and collections for Tree of Life."""
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient, ASCENDING
from app.core.config import settings


class MongoDBManager:
    """Manages Async Motor client and Sync PyMongo client."""
    def __init__(self):
        self.async_client: Optional[AsyncIOMotorClient] = None
        self.async_db: Optional[AsyncIOMotorDatabase] = None
        self.sync_client: Optional[MongoClient] = None

    def get_sync_db(self):
        if self.sync_client is None:
            self.sync_client = MongoClient(settings.MONGO_URI)
        return self.sync_client[settings.MONGO_DB_NAME]

    async def connect(self):
        if self.async_client is None:
            self.async_client = AsyncIOMotorClient(settings.MONGO_URI)
            self.async_db = self.async_client[settings.MONGO_DB_NAME]
            await self._ensure_indexes()

    async def close(self):
        if self.async_client is not None:
            self.async_client.close()
            self.async_client = None
            self.async_db = None

    async def _ensure_indexes(self):
        """Ensure indexes for fast lookups, expiration janitor, and variety matching."""
        if self.async_db is None:
            return

        # Prediction requests: index request_id and retention expiry
        await self.async_db.prediction_requests.create_index([("request_id", ASCENDING)], unique=True)
        await self.async_db.prediction_requests.create_index([("retention_expires_at", ASCENDING)])

        # Varieties and genomic profiles: index canonical_id and aliases
        await self.async_db.varieties.create_index([("canonical_id", ASCENDING)], unique=True)
        await self.async_db.varieties.create_index([("aliases", ASCENDING)])
        await self.async_db.genomic_profiles.create_index([("canonical_variety_id", ASCENDING)], unique=True)

        # Expert escalations: index request_id and status
        await self.async_db.expert_escalations.create_index([("escalation_id", ASCENDING)], unique=True)
        await self.async_db.expert_escalations.create_index([("status", ASCENDING)])
        await self.async_db.expert_escalations.create_index([("retention_expires_at", ASCENDING)])


db_manager = MongoDBManager()
