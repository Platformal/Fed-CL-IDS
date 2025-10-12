from collections.abc import Iterable
from typing import Callable
from flwr.server import ClientManager, Grid
from flwr.common import ArrayRecord, ConfigRecord, FitIns, Message
from flwr.common.typing import Parameters
from flwr.common import MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg
import json

class UAVIDSFedAvg(FedAvg):
    def __init__(self, fraction_train, fraction_eval, list_flows: list[list[int]]):
        super().__init__(fraction_train, fraction_eval)
    
    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        flows: list[list[int]] = json.loads(record['config']['flows'])
        messages = []
        for i, node_id in enumerate(node_ids):  # one message for each node
            client_record = record.copy()
            client_record['config'] = ConfigRecord({'flows': flows[i]})
            message = Message(
                content=client_record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages