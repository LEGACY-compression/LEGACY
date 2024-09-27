import torch

class Device_Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Device_Singleton, cls).__new__(cls)
        return cls._instance

    def set_device(self, id):
        self.device = torch.device("cuda:"+str(id) if torch.cuda.is_available() else "cpu")

    def get_get_device(self):
        return self.device


