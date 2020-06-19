from abc import ABC


class TurnRecord:
    def __init__(self, turn_number, turn_type, prior_z_vec, posterior_z_vec):
        self.turn_number = turn_number
        self.turn_type = turn_type
        self.prior_z_vector = prior_z_vec
        self.posterior_z_vector = posterior_z_vec

    def __str__(self):
        return f'Turn {self.turn_number}, prior {self.prior_z_vector}, posterior {self.posterior_z_vector}'


class Evaluator(ABC):
    def eval_from_dir(self, directory, role=None):
        raise NotImplementedError
