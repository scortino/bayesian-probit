class BaseSampler:
    def __repr__(self):
        attr = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__name__}({attr})"
