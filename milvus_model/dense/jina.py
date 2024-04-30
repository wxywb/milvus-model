from typing import List
from collections import defaultdict
import requests
import numpy as np

from milvus_model.base import BaseEmbeddingFunction


class JinaEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "jina-embeddings-v2-base-en",
        api_key: str,
        **kwargs,
    ):
        self._model_config = {"model": model_name}
        self.access_token = api_key

        self.model_name = model_name
        self._jinaai_model_meta_info = defaultdict(dict)
        self._jinaai_model_meta_info["jina-embeddings-v2-base-en"]["dim"] = 768
        self._jinaai_model_meta_info["jina-embeddings-v2-base-de"]["dim"] = 768
        self._jinaai_model_meta_info["jina-embeddings-v2-base-code"]["dim"] = 768
        self._jinaai_model_meta_info["jina-embeddings-v2-base-zh"]["dim"] = 768
        self._jinaai_model_meta_info["jina-embeddings-v2-base-es"]["dim"] = 768
        self._dim = self._jinaai_model_meta_info[model_name]["dim"]
        
    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents)

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    @property
    def dim(self):
        return self._dim

    def _encode_query(self, query: str) -> np.array:
        return self._encode([query])[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode([document])[0]

    def _call_jina_api(self, texts: List[str]):
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        data = {
            'input': texts,
            'model': self._model_config["model"]
        }
        response = requests.post(url, headers=headers, json=data)
        obj = response.json()
        return [np.array(result['embedding']) for result in obj['data']]

    def _encode(self, texts: List[str]):
        return self._call_jina_api(texts)

