import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.config.get(key, None)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __repr__(self):
        return str(self.config)
