from messages import AggregateMsg, UpdateMsg, InitMsg
import json, os, copy, time


class state:
    def __init__(self, round_, agg_gradient, base_gradient, base_result, aggregation_timeout=3):
        self.round = round_  # this round
        self.agg_gradient = agg_gradient  # last round's incoming-models that aggregate to new base-model
        self.gradient_result = base_gradient
        self.base_result = base_result  # base-model that aggregate from last round's incoming-models
        self.incoming_gradient = []  # collection from this round
        self.selection_nonce = 0
        self.aggregation_timeout = aggregation_timeout  # 3 block
        self.aggregation_timeout_count = 0
        self.number_of_validator = 4
        self.aggregator_id = -1
        self.train_lr = -1.0

    def json(self):
        return json.dumps(self.__dict__)


class State_controller:
    def __init__(self, logger, trainer, aggregator, threshold):
        self.logger = logger
        self.states = []

        self.trainer = trainer
        self.aggregator = aggregator
        self.threshold = threshold

        self.max_iteration = 50
        self.is_saved = False

        self.aggregation_lock = False

        self.model_list = []

    def aggregate_pipe(self, tx):
        data = AggregateMsg(**tx)
        if self.get_last_round() == data.get_round():
            self.logger.info("Get round : {} ".format(data.get_round()))
            self.logger.info("round exist, now at round : {} ".format(self.get_last_round()))
            return
        if not str(self.get_last_state()["aggregator_id"]) == str(data.get_cid()):
            self.logger.info(
                "Invalid aggregate cid, the aggregator id should be {}".format(self.get_last_state()["aggregator_id"]))
            return
        new_base = self.trainer.opt_step_base_model(txmanager=self, base_gradient=data.get_result(), round_ = self.get_last_round())
        self.model_list.append(copy.deepcopy(new_base))
        state_data = state(round_=data.get_round(),
                           agg_gradient=data.get_weight(),
                           base_gradient=data.get_result(),
                           base_result=len(self.model_list)-1,
                           aggregation_timeout=self.get_last_state()['aggregation_timeout'])
        self.states.append(eval(state_data.json()))
        self.aggregation_lock = False
        # make a point save here
        self.save_round_point(data.get_round())

    def update_pipe(self, tx):
        data = UpdateMsg(**tx)
        # data.get_round
        if self.aggregation_lock:
            return
        if self.get_last_round() == data.get_round():
            self.append_incoming_gradient({"gradient": data.get_weight(), "cid": data.get_cid()})
            self.logger.info("Get incoming gradient, round: {}, total: {}".format(self.get_last_round(),
                                                                                  len(self.get_incoming_gradient())))

    def create_task_pipe(self, tx):
        data = InitMsg(**tx)
        self.model_list.append(copy.deepcopy(self.trainer.get_model_by_ipfs(data.get_weight())))
        state_data = state(round_=0,
                           agg_gradient=[],
                           base_gradient="0000",
                           base_result=len(self.model_list)-1,
                           aggregation_timeout=data.get_agg_timeout())
        self.states.append(eval(state_data.json()))
        self.trainer.train_run(data.get_weight(), 0)
        self.max_iteration = data.get_max_iteration()
        self.logger.info("Max iteration: {}".format(data.get_max_iteration()))
        # make a point save here
        self.save_round_point(0)

    def pipes(self, type_):
        dis = {"create_task": self.create_task_pipe, "aggregation": self.aggregate_pipe, "update": self.update_pipe}
        return dis[type_]

    #######################################################
    def tx_manager(self, tx):
        if not self.task_end_check():
            return

        if tx is None and self.aggregation_lock:  # Endblock : tx = None
            self.get_last_state()["aggregation_timeout_count"] += 1
            if self.get_last_state()["aggregation_timeout_count"] >= self.get_last_state()["aggregation_timeout"]:
                self.get_last_state()["aggregation_timeout_count"] = 0
                self.get_last_state()["selection_nonce"] += 1
                self.aggregator.aggergate_manager(txmanager=self, tx={"type": "aggregate_again"})
            return
        if tx is None:
            return
        self.pipes(tx["type"])(tx)

        self.trainer.train_manager(txmanager=self, tx=tx)
        self.aggregator.aggergate_manager(txmanager=self, tx=tx)

    #######################################################
    def tx_checker(self, tx) -> bool:
        # self.logger.info(tx)
        try:
            if tx["type"] == "aggregation":
                _ = AggregateMsg(**tx)
            elif tx["type"] == "update":
                _ = UpdateMsg(**tx)
            elif tx["type"] == "create_task":
                # self.logger.info(">> create_task")
                _ = InitMsg(**tx)
            self.logger.info("TX valid.")
            return True
        except KeyError:
            self.logger.info("TX invalid.")
            return False

    def get_last_round(self):
        try:
            round_ = self.states[-1]["round"]  # init state have no element
        except:
            round_ = -1
        return round_

    # def set_last_base_model(self, model):
    #     self.states[-1]["base_result"] = model

    def get_last_base_model(self):
        try:
            state_ = self.model_list[self.states[-1]["base_result"]]  # init state have no element
        except:
            state_ = ""
        return state_

    def get_last_gradient_result(self):
        try:
            state_ = self.states[-1]["gradient_result"]  # init state have no element
        except:
            state_ = ""
        return state_

    def append_incoming_gradient(self, value):
        self.states[-1]["incoming_gradient"].append(value)

    def get_incoming_gradient(self):
        return [i["gradient"] for i in self.states[-1]["incoming_gradient"]]

    def get_last_nonce(self):
        try:
            return self.states[-1]["selection_nonce"]
        except:
            return 0

    def set_last_nonce(self, value):
        try:
            self.states[-1]["selection_nonce"] = value
        except:
            pass
    
    def set_last_lr(self, value):
        try:
            self.states[-1]["train_lr"] = value
        except:
            pass

    def get_last_state(self):
        try:
            return self.states[-1]
        except:
            return 0

    def task_end_check(self) -> bool:
        # self.logger.info(">>>>>>>> {}".format(len(self.states)))
        # self.logger.info(">>>>>>>> {}".format(self.get_last_round()))
        if len(self.states) >= self.max_iteration and self.get_last_round() >= self.max_iteration - 1:
            if not self.is_saved:
                time.sleep(30)
                # save model
                states_ = copy.deepcopy(self.states)
                if int(os.getenv("ID")) == 0:
                    import torch
                    if not os.path.exists("/root/py-app/save_models/"):
                        os.mkdir("/root/py-app/save_models/")

                    for i in states_:
                        model_save = "/root/py-app/save_models/round_{}.pt".format(i["round"])
                        torch.save(self.model_list[i["base_result"]].state_dict(), model_save)
                        i["base_result"] = "round_{}.pt".format(i["round"])
                        for j in i["incoming_gradient"]:
                            model_save = "/root/py-app/save_models/round_{}_cid_{}.pt".format(i["round"], j["cid"])
                            cid_model = self.trainer.opt_step_base_model(txmanager=self, base_gradient=j["gradient"], round_=i["round"])
                            torch.save(cid_model.state_dict(), model_save)

                # save json report
                result = {"data": states_}
                with open('/root/py-app/{}_round_result_{}.json'.format(self.max_iteration, os.getenv("ID")),
                          'w') as outfile:
                    json.dump(result, outfile, indent=4)
                self.logger.info("Save to file....")
            self.is_saved = True
            # Done
            if int(os.getenv("ID")) == 0:
                open("/root/py-app/save/Done", 'a').close()
            return False
        else:
            return True

    def save_round_point(self, round_):
        if not os.path.exists("/root/py-app/save/"):
            os.mkdir("/root/py-app/save/")
        open("/root/py-app/save/round_{}".format(round_), 'a').close()
