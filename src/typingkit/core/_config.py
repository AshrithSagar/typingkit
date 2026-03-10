"""
Runtime validation configuration
=======
"""
# src/typingkit/core/_config.py

__all__ = [
    "TypedCollectionConfig",
]


class TypedCollectionConfig:
    """Global switches for runtime validation in TypedList and TypedDict."""

    VALIDATE_LENGTH: bool = True
    VALIDATE_ITEM: bool = True

    @classmethod
    def enable_all(cls) -> None:
        cls.VALIDATE_LENGTH = True
        cls.VALIDATE_ITEM = True

    @classmethod
    def disable_all(cls) -> None:
        cls.VALIDATE_LENGTH = False
        cls.VALIDATE_ITEM = False
