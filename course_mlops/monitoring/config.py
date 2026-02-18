from pydantic import BaseModel

from course_mlops.utils import EnvironmentVariable


class MonitoringConfig(BaseModel):
    database_url: str
    drift_window_size: int = 500
    drift_min_samples: int = 50
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        return cls(
            database_url=f"postgresql+psycopg://{EnvironmentVariable.DB_USER.read()}:{EnvironmentVariable.DB_PASSWORD.read()}@{EnvironmentVariable.DB_HOST.read()}:{EnvironmentVariable.DB_PORT.read()}/{EnvironmentVariable.DB_NAME.read()}",
            drift_window_size=int(EnvironmentVariable.DRIFT_WINDOW_SIZE.read(default="500")),
            drift_min_samples=int(EnvironmentVariable.DRIFT_MIN_SAMPLES.read(default="50")),
        )
