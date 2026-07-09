from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    LOG_LEVEL: str = "INFO"
    SQL_ECHO: bool = False

    # Error-Tracking (leer = deaktiviert)
    SENTRY_DSN: str = ""

    # Frontend
    FRONTEND_URL: str
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # Stripe Config
    STRIPE_SECRET_KEY: str
    STRIPE_WEBHOOK_SECRET: str
    # Erst aktivieren, wenn Stripe Tax im Dashboard eingerichtet ist
    STRIPE_AUTOMATIC_TAX: bool = False
    STRIPE_PRICE_ID_PRO_MONTHLY: str
    STRIPE_PRICE_ID_PRO_YEARLY: str
    STRIPE_SUCCESS_URL: str
    STRIPE_CANCEL_URL: str

    # Email Config
    EMAIL_FROM: str
    SMTP_HOST: str
    SMTP_PORT: int
    SMTP_USER: str
    SMTP_PASSWORD: str
    # Empfänger-Adresse für das Support-Kontaktformular (frontend/src/pages/legal/ContactPage.tsx
    # und /app/support) - konfigurierbar statt hartkodiert, da sie sich vom SMTP-Absender
    # (EMAIL_FROM) unterscheiden kann.
    SUPPORT_EMAIL: str = "gecenanalysis@gmail.com"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()