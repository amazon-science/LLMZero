class BaseAgent:
    def __init__(self, config, manager):
        self.config = config
        self.manager = manager

    @property
    def training_framework(self):
        from ..training_frameworks import get_framework_from_config

        return get_framework_from_config(self.config)
