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
    
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        messages = super().configure_train(server_round, arrays, config, grid)
        flows: list[list[int]] = json.loads(config['flows'])
        for i, message in enumerate(messages):
            client_content = message.content.copy()
            client_content['config'] = client_content['config'].copy()
            client_content['config']['flows'] = flows[i]
            message.content = client_content
        return messages
    
    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        messages = super().configure_evaluate(server_round, arrays, config, grid)
        flows: list[list[int]] = json.loads(config['flows'])
        for i, message in enumerate(messages):
            client_content = message.content.copy()
            client_content['config'] = client_content['config'].copy()
            client_content['config']['flows'] = flows[i]
            message.content = client_content
        return messages
        
    # def _construct_messages(
    #     self, record: RecordDict, node_ids: list[int], message_type: str
    # ) -> Iterable[Message]:
    #     print(record['config'].keys())
    #     flows: list[list[int]] = json.loads(record['config']['flows'])
    #     messages = []
    #     for i, node_id in enumerate(node_ids):  # one message for each node
    #         client_record = record.copy()
    #         client_record['config'] = ConfigRecord({'flows': flows[i]})
    #         message = Message(
    #             content=client_record,
    #             message_type=message_type,
    #             dst_node_id=node_id,
    #         )
    #         messages.append(message)
    #     return messages