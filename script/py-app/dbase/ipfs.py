import ipfshttpclient
import time
from abci.utils import get_logger

log = get_logger()


class ipfs:
    def __init__(self, addr="/ip4/172.168.10.10/tcp/5001/http"):
        while True:
            try:
                self.client = ipfshttpclient.connect(addr)
                break
            except:
                log.info("Waiting for ipfs services at : ", addr)
                time.sleep(1)

    def add(self, data):
        return self.client.add_str(data)

    def cat(self, data):
        return self.client.cat(data).decode()
