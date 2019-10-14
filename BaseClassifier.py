from abc import ABC, abstractmethod

import pandas as pd


class SentimentClassifier(ABC):

    @abstractmethod
    def classify_df(self, twitch_data: pd.DataFrame) -> pd.DataFrame:
        pass
