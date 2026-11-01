import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    negotiator_model: str = "openai/gpt-oss-20b"
    mediator_model: str = "openai/gpt-oss-20b"
    max_rounds: int = 10
    max_retries: int = 3
    temperature: float = 0.7
    mediator_temperature: float = 0.4
    concession_decay: float = 0.85
    impasse_threshold: float = 0.01
    impasse_patience: int = 3
    base_dir: Path = Path(__file__).resolve().parent.parent

    _custom_data_dir: Path | None = None
    _custom_chroma_dir: Path | None = None
    _custom_sqlite_path: Path | None = None

    @property
    def data_dir(self) -> Path:
        if self._custom_data_dir is not None:
            return Path(self._custom_data_dir)
        if os.environ.get("VERCEL"):
            return Path("/tmp/consensus_data")
        return self.base_dir / "data"

    @data_dir.setter
    def data_dir(self, val: Path) -> None:
        self._custom_data_dir = Path(val)

    @property
    def chroma_dir(self) -> Path:
        if self._custom_chroma_dir is not None:
            return Path(self._custom_chroma_dir)
        return self.data_dir / "chroma"

    @chroma_dir.setter
    def chroma_dir(self, val: Path) -> None:
        self._custom_chroma_dir = Path(val)

    @property
    def sqlite_path(self) -> Path:
        if self._custom_sqlite_path is not None:
            return Path(self._custom_sqlite_path)
        return self.data_dir / "sqlite" / "consensus.db"

    @sqlite_path.setter
    def sqlite_path(self, val: Path) -> None:
        self._custom_sqlite_path = Path(val)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "allow"}


settings = Settings()
