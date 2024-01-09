class CaseInsensitiveDict(dict):
    def __getitem__(self, key):
        key = key.lower()
        for k in self.keys():
            if k.lower() == key:
                key = k
                break
        return super().__getitem__(key)
