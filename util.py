from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    openai_api_key: str = Field()


settings = Settings()


model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
)
