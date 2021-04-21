import json
import uuid


class BaseMsg:
    __msg_type = ""
    __msg_cid = 0
    __msg_round = 0

    def __init__(self, **kwargs):
        self._type = kwargs["type"] if "type" in kwargs else BaseMsg.__msg_type
        self._cid = kwargs["cid"] if "cid" in kwargs else BaseMsg.__msg_cid
        self._round = kwargs["round"] if "round" in kwargs else BaseMsg.__msg_round

    def get_type(self):
        return self._type

    def get_cid(self):
        return self._cid

    def get_round(self):
        return self._round

    def set_type(self, value):
        self._type = value

    # @classmethod
    def set_cid(cls, value):
        cls._cid = value

    def set_round(self, value):
        self._round = value

    # def __str__(self):
    #     tmp = self.__dict__
    #     return json.dumps(tmp)
    #
    # def json_serialize(self):
    #     return self.__str__()


class UpdateMsg(BaseMsg):
    __msg_weight = ""

    def __init__(self, **kwargs):
        if not "type" in kwargs:
            kwargs["type"] = "update"
        else:
            if not kwargs["type"] == "update":
                raise TypeError(f'TypeError(update)<={kwargs["type"]}')
        super().__init__(**kwargs)
        self._weight = kwargs["weight"] if "weight" in kwargs else UpdateMsg.__msg_weight

    def get_weight(self):
        return self._weight

    def set_weight(self, value):
        self._weight = value

    def __str__(self):
        output = {}
        data = self.__dict__
        for k, v in data.items():
            output[k[1:]] = v
        return json.dumps(output)

    def json_serialize(self):
        return self.__str__()


class AggregateMsg(BaseMsg):
    __msg_weight = []
    __msg_result = ""

    def __init__(self, **kwargs):
        if not "type" in kwargs:
            kwargs["type"] = "aggregation"
        else:
            if not kwargs["type"] == "aggregation":
                raise TypeError(f'TypeError(aggregation)<={kwargs["type"]}')

        super().__init__(**kwargs)
        self._weight = kwargs["weight"] if "weight" in kwargs else AggregateMsg.__msg_weight
        self._result = kwargs["result"] if "result" in kwargs else AggregateMsg.__msg_result

    def get_weight(self):
        return self._weight

    def get_result(self):
        return self._result

    def set_weight(self, value):
        self._weight = value

    def set_result(self, value):
        self._result = value

    def __str__(self):
        output = {}
        data = self.__dict__
        for k, v in data.items():
            output[k[1:]] = v
        return json.dumps(output)

    def json_serialize(self):
        return self.__str__()


class InitMsg:
    __msg_type = ""
    __msg_weight = ""
    __msg_max_iteration = ""
    __msg_agg_timeout = 3

    def __init__(self, **kwargs):
        if not "type" in kwargs:
            kwargs["type"] = "create_task"
        else:
            if not kwargs["type"] == "create_task":
                raise TypeError(f'TypeError(create_task)<={kwargs["type"]}')

        self._type = kwargs["type"] if "type" in kwargs else InitMsg.__msg_type
        self._weight = kwargs["weight"] if "weight" in kwargs else InitMsg.__msg_weight
        self._max_iteration = kwargs["max_iteration"] if "max_iteration" in kwargs else InitMsg.__msg_max_iteration
        self._agg_timeout = kwargs["aggtimeout"] if "aggtimeout" in kwargs else InitMsg.__msg_agg_timeout

    def get_weight(self):
        return self._weight

    def get_max_iteration(self):
        return self._max_iteration

    def get_agg_timeout(self):
        return self._agg_timeout

    def set_weight(self, value):
        self._weight = value

    def set_max_iteration(self, value):
        self._max_iteration = value

    def set_agg_timeout(self, value):
        self._agg_timeout = value

    def __str__(self):
        output = {}
        data = self.__dict__
        for k, v in data.items():
            output[k[1:]] = v  # remove "_" before key
        return json.dumps(output)

    def json_serialize(self):
        return self.__str__()


if __name__ == "__main__":
    a = UpdateMsg()
    print(a.json_serialize())

    b = AggregateMsg()
    b.set_round(23)
    b.set_weight(["abc", "def"])
    b.set_result("123")
    print(b.json_serialize())
    # > {"type": "aggregation", "cid": 0, "task": "8711c4a6b0c34330a5225abf0fd25bb8",
    #    "round": 23, "weight": ["abc", "def"], "result": "123"}

    c = InitMsg()
    c.set_sample(1)
    c.set_weight("123456")
    c.set_max_iteration(100)
    print(c.json_serialize())
    # > {"type": "create_task", "weight": "123456", "max_iteration": 100, "sample": 1}

    d = InitMsg(**json.loads(c.json_serialize()))
    print(d)
    # > {"type": "create_task", "weight": "123456", "max_iteration": 100, "sample": 1}

    data = {'type': 'create_task', 'max_iteration': 100, 'sample': 0.5, 'weight': '12346'}
    e = InitMsg(**data)
    print(e)

    f = {"type": "aggregation", "cid": "3", "round": 1, "weight": ["83239157", "43887821", "38935139", "33500173"],
         "result": "52071330"}
    g = AggregateMsg(**f)
    print(g.json_serialize())
