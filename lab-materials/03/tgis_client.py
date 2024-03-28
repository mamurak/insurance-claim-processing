from re import sub

from tgis_grpc_client.tgis_grpc_client import TgisGrpcClient


class TgisClient:
    def __init__(
            self, infer_endpoint, model_id='flan-t5-small',
            grpc_port=443, max_tokens=256, verbose=False):

        self.model_id = model_id
        self.max_tokens = 256

        hostname = sub("https://|http://", "", infer_endpoint)
        if hostname[-1] == "/":
            hostname = hostname[:-1]
        self._client = TgisGrpcClient(
            hostname, grpc_port, verify=False, verbose=verbose
        )
        
    def predict(self, text, query):
        prompt = f'{text}\n{query}'
        raw_response = self._client.make_request(
            prompt, model_id=self.model_id, max_new_tokens=self.max_tokens
        )
        response = raw_response.responses[0].text
        return response
