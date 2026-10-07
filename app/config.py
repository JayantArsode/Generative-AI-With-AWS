import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from app.exceptions import ProviderConfigError
from app.schemas.provider_config import ProviderConfig, ProvidersConfig

load_dotenv()

PROVIDERS_FILE = Path(__file__).resolve().parent.parent / "providers.yml"


def load_providers(file_path: Path = PROVIDERS_FILE) -> ProvidersConfig:
    """
    Load and validate the providers config file.

    Parameters
    ----------
    file_path : Path, optional
        Path of the YAML file, by default ``providers.yml`` in the project root.

    Returns
    -------
    ProvidersConfig
        All providers and the name of the default one.

    Raises
    ------
    ProviderConfigError
        If the file is missing, is not valid YAML, or has invalid settings.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ProviderConfigError(f"Providers config file not found: {file_path}")
    except yaml.YAMLError as ye:
        raise ProviderConfigError(f"Error parsing {file_path}: {ye}")

    try:
        return ProvidersConfig.model_validate(data)
    except ValidationError as ve:
        raise ProviderConfigError(f"Invalid providers config in {file_path}:\n{ve}")


def get_provider(
    name: str | None = None, file_path: Path = PROVIDERS_FILE
) -> ProviderConfig:
    """
    Get the settings of one provider.

    Parameters
    ----------
    name : str | None, optional
        Name of the provider. ``None`` uses the default from the file.
    file_path : Path, optional
        Path of the YAML file, by default ``providers.yml`` in the project root.

    Returns
    -------
    ProviderConfig
        The provider's settings.

    Raises
    ------
    ProviderConfigError
        If the file is invalid or there is no provider with that name.
    """
    config = load_providers(file_path)
    name = name or config.default

    if name not in config.providers:
        raise ProviderConfigError(
            f"Unknown provider '{name}'. Choose one of: {', '.join(config.providers)}"
        )
    return config.providers[name]


def get_api_key(provider: ProviderConfig) -> str | None:
    """
    Read the provider's API key from the environment or ``.env``.

    Parameters
    ----------
    provider : ProviderConfig
        The provider whose key to read.

    Returns
    -------
    str | None
        The API key, or ``None`` if the provider needs no key.

    Raises
    ------
    ProviderConfigError
        If the provider needs a key and the variable is not set.
    """
    if provider.api_key_env is None:
        return None

    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        raise ProviderConfigError(
            f"Provider '{provider.name}' needs an API key. "
            f"Set {provider.api_key_env} in .env"
        )
    return api_key
