import os

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    openai_api_key: str = Field()

    langfuse_secret_key: str = Field(default=None)
    langfuse_public_key: str = Field(default=None)
    langfuse_host: str = Field(default=None)


settings = Settings()

# to set env vars - used by langfuse
load_dotenv()

# Print those env vars:
print("LANGFUSE_PUBLIC_KEY:", os.getenv("LANGFUSE_PUBLIC_KEY"))
# print("LANGFUSE_SECRET_KEY:", os.getenv("LANGFUSE_SECRET_KEY"))
print("LANGFUSE_HOST:", os.getenv("LANGFUSE_HOST"))

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
)
