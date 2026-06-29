from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

from ..models.issue import Issue
from ..config import Config


class BaseAnalyzer(ABC):
    """Every analyzer takes a Config, looks at dataframe(s), and returns a
    list of Issue objects. No printing, no plotting, no scoring inside an
    analyzer — that all happens elsewhere."""

    name: str = "base"

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def run(
        self,
        train: pd.DataFrame,
        test: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
    ) -> List[Issue]:
        ...
