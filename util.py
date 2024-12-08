from autogen_ext.models._openai._openai_client import OpenAIChatCompletionClient
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field()


settings = Settings()


model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
)
