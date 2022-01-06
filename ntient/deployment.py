from .base import Base
from .model import Model


class Deployment(Base):
    def __init__(self,
                 model: Model,
                 environment: str = "",
                 instances: int = 1,
                 v_cores: int = 1
                 ):

        self.model = model
        self.environment = environment
        self.instances = environment
        self.v_cores = v_cores
        self.instances = instances
