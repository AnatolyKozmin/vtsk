from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    db_user: str = "vtsk"
    db_password: str = "1234"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "vtsk_db"
    
    sender_host: str = "0.0.0.0"
    sender_port: int = 5000
    receiver_host: str = "0.0.0.0"
    receiver_port: int = 5001
    
    receiver_url: str = "http://127.0.0.1:5001"
    sender_callback_url: str = "http://127.0.0.1:5000"
    
    max_workers: int = 100
    request_timeout: float = 30.0
    
    zabbix_url: Optional[str] = None
    zabbix_user: Optional[str] = None
    zabbix_password: Optional[str] = None
    pfsense_host_name: str = "pfsense"
    waf_host_name: str = "nemesida-waf"
    
    sla_availability_target: float = 99.9
    sla_latency_p95_target: float = 200.0
    sla_latency_p99_target: float = 500.0
    sla_throughput_target: float = 1000.0
    
    statistical_confidence_level: float = 0.95
    anomaly_detection_window: int = 100
    baseline_collection_period: int = 300
    
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
