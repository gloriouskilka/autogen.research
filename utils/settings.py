from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    openai_api_key: str = Field()

    langfuse_secret_key: str = Field(default=None)
    langfuse_public_key: str = Field(default=None)
    langfuse_host: str = Field(default=None)

    weatherstack_api_key: str = Field(default=None)


settings = Settings()
