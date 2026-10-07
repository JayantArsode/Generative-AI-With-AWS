class ProviderConfigError(Exception):
    """
    Raised when the providers config file is missing, invalid, or points to
    an API key variable that is not set.
    """
