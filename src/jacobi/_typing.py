from typing import Sized, Generic, TypeVar
import abc

T = TypeVar("T")


class Indexable(Sized, Generic[T]):
    """Indexable type for mypy."""

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> T:
        """Get item at index idx."""
        ...  # pragma: no cover
