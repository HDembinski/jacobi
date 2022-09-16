from typing import Iterable, Sized, Generic, TypeVar

T = TypeVar("T")


class Indexable(Iterable, Sized, Generic[T]):
    """Indexable type for mypy."""

    def __getitem__(self, idx: int) -> T:
        """Get item at index idx."""
        ...  # pragma: no cover
