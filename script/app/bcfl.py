import struct
import abci.utils as util
import argparse
import json, os

from abci import (
    ABCIServer,
    BaseApplication,
    ResponseInfo,
    ResponseInitChain,
    ResponseCheckTx,
    ResponseDeliverTx,
    ResponseQuery,
    ResponseCommit,
    CodeTypeOk,
    ResponseEndBlock,
)

from aggregator import aggregator
from trainer import trainer
from db import db as moddb
from tx_handler import tx as sender
from state_controller import State_controller

from options import Configer

log = util.get_logger()


def encode_number(value):
    return struct.pack('>I', value)


def decode_number(raw):
    return int.from_bytes(raw, byteorder='big')


class SimpleBCFL(BaseApplication):
    def __init__(self, controller):
        self.controller = controller

    def info(self, req) -> ResponseInfo:
        """
        Since this will always respond with height=0, Tendermint
        will resync this app from the begining
        """
        r = ResponseInfo()
        r.version = "1.0"
        r.last_block_height = 0
        r.last_block_app_hash = b''
        return r

    def init_chain(self, req) -> ResponseInitChain:
        """Set initial state on first run"""
        log.info("Got InitChain")
        self.txCount = 0
        self.last_block_height = 0
        return ResponseInitChain()

    def check_tx(self, tx) -> ResponseCheckTx:
        """
        Validate the Tx before entry into the mempool
        Checks the txs are submitted in order 1,2,3...
        If not an order, a non-zero code is returned and the tx
        will be dropped.
        """
        log.info("Got ChectTx  {}".format(tx))
        value = eval(tx.decode())
        # log.info(value)
        if not self.controller.tx_checker(value):
            return ResponseCheckTx(code=1)  # reject code != 0
        log.info("Check ok")
        # if data["Type"] == "aggregation":
        #     if self.aggregator.aggergateCheck(data["weight"]):
        #         return ResponseCheckTx(code=CodeTypeOk)

        return ResponseCheckTx(code=CodeTypeOk)

    def deliver_tx(self, tx) -> ResponseDeliverTx:
        """Simply increment the state"""
        # value = decode_number(tx)
        # self.txCount += 1
        # log.info("Got DeliverTx {}, so txCount increase to {}".format(tx))
        log.info("Got DeliverTx  {}".format(tx))
        value = eval(tx.decode())
        # log.info(value)

        self.controller.tx_manager(value)
        log.info("Delivery ok")

        return ResponseDeliverTx(code=CodeTypeOk)

    def query(self, req) -> ResponseQuery:
        """Return the last tx count"""
        v = encode_number(self.txCount)
        return ResponseQuery(code=CodeTypeOk, value=v, height=self.last_block_height)

    def commit(self) -> ResponseCommit:
        """Return the current encode state value to tendermint"""
        log.info("Got Commit")
        hash = struct.pack('>Q', self.txCount)
        return ResponseCommit(data=hash)

    # def begin_block(self, req) -> ResponseBeginBlock:
    #     # https://pkg.go.dev/github.com/tendermint/tendermint@v0.34.3/proto/tendermint/types#Header
    #     last_hash = req.Header.LastCommitHash
    #     log.info("Last hash  {}".format(last_hash))
    #     return ResponseBeginBlock()
    def end_block(self, req) -> ResponseEndBlock:
        self.controller.tx_manager(None)
        return ResponseEndBlock()


if __name__ == '__main__':
    # Define argparse argument for changing proxy app port
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=26658, help='Proxy app port')
    parser.add_argument('-dataset', type=str, default=None, help='Path to dataset')
    parser.add_argument('-device', type=str, default="CPU", help='Device')
    parser.add_argument('-config', type=str, default=None, help='config')
    args = parser.parse_args()

    if args.config is None:
        exit("No config.ini found.")

    con = Configer(args.config)

    os.environ["DATASET"] = con.trainer.get_dataset()

    # newsender = sender(log, url_="http://node0:26657")
    newsender = sender(log, url_=con.bcfl.get_sender())
    # newdb = moddb("ipfs")
    newdb = moddb(con.bcfl.get_db())

    newagg = aggregator(log, con, newdb, newsender)
    # newtrain = trainer(log, args.dataset, newdb, newsender, devices=args.device)
    newtrain = trainer(log, con, newdb, newsender)

    newcontroller = State_controller(log, newtrain, newagg, con.agg.get_threshold())

    # Create the app
    # app = ABCIServer(app=SimpleBCFL(newcontroller), port=args.p)
    app = ABCIServer(app=SimpleBCFL(newcontroller), port=con.bcfl.get_app_port())
    # Run it
    app.run()
