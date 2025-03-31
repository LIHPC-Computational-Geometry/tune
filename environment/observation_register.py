
class ObservationRegistry:
    def __init__(self, n_darts_selected, deep, lowest_value, highest_value):
        self.n_darts = n_darts_selected
        self.deep = deep
        self.low = lowest_value
        self.high = highest_value
        self.counts = {}

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
        if self.counts.get(ID) is None:
            self.counts[ID] = 1
        else:
            self.counts[ID] += 1

    def get_count(self, observation):
        """
        Returns the number of times an observation has been seen by the agent.
        :param observation:
        :return: counts of observations
        """
        ID = self.encode(observation)
        return self.counts.get(ID, 0)

