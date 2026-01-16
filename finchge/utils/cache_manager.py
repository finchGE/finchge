from abc import ABC, abstractmethod
import diskcache as dc
from collections import OrderedDict


class CacheInterface(ABC):
    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def set(self, key, value):
        pass


class LRUCache:
    def __init__(self, maxsize=128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


class DiskCache(CacheInterface):
    def __init__(self, cache_dir='cache', size_limit=2**30):  # 1GB limit by default
        self.cache = dc.Cache(cache_dir, size_limit=size_limit)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache.set(key, value)

    def clear(self):
        self.cache.clear()


class CacheManager:
    def __init__(self, cache_type='lru', **kwargs):
        if cache_type == 'lru':
            self.cache = LRUCache(**kwargs)
        elif cache_type == 'disk':
            self.cache = DiskCache(**kwargs)
        else:
            raise ValueError("Unsupported cache type: {}".format(cache_type))

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache.set(key, value)

    def clear(self):
        self.cache.clear()
