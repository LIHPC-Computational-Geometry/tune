import pandas as pd


class ObservationRegistry:
    def __init__(self, n_darts_selected, deep, lowest_value, highest_value):
        self.n_darts = n_darts_selected
        self.deep = deep
        self.low = lowest_value
        self.high = highest_value
        self.df = pd.DataFrame(columns=["counts"])
        self.df.index.name = "observations"

    def encode(self, observation):
        """
        Converts an observation into a tuple.
        :param observation:
        :return: the tuple ID of the observation
        """
        ID = tuple(tuple(dart_surrounding) for dart_surrounding in observation)
        return ID

    def decode(self, ID):
        """
        Reconstructs an observation from its ID.
        :param ID: a tuple ID
        :return: the observation matrix
        """
        return [list(dart_surrounding) for dart_surrounding in ID]

    def register_observation(self, observation) -> None:
        """
        Adds an observation to the register and increments its counter.
        :param observation:
        """
        ID = self.encode(observation)

        if self.df.empty:
            self.df = pd.DataFrame({"counts": [1]}, index=[ID])
        elif ID in self.df.index:
            self.df.at[ID, "counts"] += 1
        else:
            new_row = pd.DataFrame({"counts": [1]}, index=[ID])
            self.df = pd.concat([self.df, new_row])

    def get_count(self, observation):
        """
        Returns the number of times an observation has been seen by the agent.
        :param observation:
        :return: counts of observations
        """
        ID = self.encode(observation)
        return int(self.df.loc[ID])

    def save(self, path):
        if path.endswith(".csv"):
            self.df.to_csv(path)
        elif path.endswith(".parquet"):
            self.df.to_parquet(path)
        else:
            print("Unsupported file type")

    def load_counts(self, path):
        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        elif path.endswith(".parquet"):
            self.df = pd.read_parquet(path)
        else:
            print("Unsupported file type")