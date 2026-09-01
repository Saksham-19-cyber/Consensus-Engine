from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    negotiator_model: str = "llama-3.3-70b-versatile"
    mediator_model: str = "llama-3.1-8b-instant"
    max_rounds: int = 10
    max_retries: int = 3
    temperature: float = 0.7
    mediator_temperature: float = 0.4
    concession_decay: float = 0.85
    impasse_threshold: float = 0.01
    impasse_patience: int = 3
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    chroma_dir: Path = data_dir / "chroma"
    sqlite_path: Path = data_dir / "sqlite" / "consensus.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
