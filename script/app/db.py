import time
import uuid
from abci.utils import get_logger
from dbase.ipfs import ipfs

MODE = "ipfs"

if MODE == "ipfs":
    import ipfshttpclient

log = get_logger()


class db:
    def __init__(self, mode=["ipfs"]):
        if "ipfs" in mode:
            self.db_ = ipfs(addr="/ip4/172.168.10.10/tcp/5001/http")

    def add(self, data):
        return self.db_.add(data)

    def cat(self, data):
        return self.db_.cat(data)
