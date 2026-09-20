"""AURORA — shared contracts for the In-Vehicle AI Assistant platform."""

from .version import __version__ as __version__  # noqa: F401
from .version import PROMPT_VERSION as PROMPT_VERSION  # noqa: F401
from .version import RAG_INDEX_VERSION as RAG_INDEX_VERSION  # noqa: F401
from .version import TELEMETRY_SCHEMA_VERSION as TELEMETRY_SCHEMA_VERSION  # noqa: F401

PLATFORM_NAME = "AURORA"
PLATFORM_TAGLINE = "Production In-Vehicle AI Assistant MLOps Platform"