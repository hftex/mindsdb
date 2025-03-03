import io
import json

import requests
from typing import Dict, Optional

import pandas as pd
import pyarrow.parquet as pq

from mindsdb.integrations.libs.base import BaseMLEngine

def dictWithoutNone (dict):
    return {k: v for k, v in dict.items() if v is not None}

class RayServeException(Exception):
    pass


class RayServeHandler(BaseMLEngine):
    """
    The Ray Serve integration engine needs to have a working connection to Ray Serve. For this:
        - A Ray Serve server should be running

    Example:

    """  # noqa
    name = 'ray_serve'

    @staticmethod
    def create_validation(target, args=None, **kwargs):
        if not args.get('using'):
            raise Exception("Error: This engine requires some parameters via the 'using' clause. Please refer to the documentation of the Ray Serve handler and try again.")  # noqa
        if not args['using'].get('train_url'):
            raise Exception("Error: Please provide a URL for the training endpoint.")
        if not args['using'].get('predict_url'):
            raise Exception("Error: Please provide a URL for the prediction endpoint.")

    def create(self, target: str, df: Optional[pd.DataFrame] = None, args: Optional[Dict] = None) -> None:
        # TODO: use join_learn_process to notify users when ray has finished the training process
        print("*** create args", args)
        args = args['using']  # ignore the rest of the problem definition
        args['target'] = target
        self.model_storage.json_set('args', args)
        try:
            if args.get('is_parquet', False):
                buffer = io.BytesIO()
                df.to_parquet(buffer)
                resp = requests.post(args['train_url'],
                                     files={"df": ("df", buffer.getvalue(), "application/octet-stream")},
                                     data={"args": json.dumps(args), "target": target},
                                     )
            else:
                resp = requests.post(args['train_url'],
                                     json={'df': df.to_json(orient='records'), 'target': target, 'args': args},
                                     headers={'content-type': 'application/json; format=pandas-records'})
        except requests.exceptions.InvalidSchema:
            raise Exception("Error: The URL provided for the training endpoint is invalid.")

        error = None
        try:
            resp = resp.json()
        except json.JSONDecodeError:
            error = resp.text
        else:
            if resp.get('status') != 'ok':
                error = resp['status']

        if error:
            raise RayServeException(f"Error: {error}")


    # def predict(self, df, args=None):
    #     print("*** predict incoming args", args)
    #     # e.g. {'pred_format': 'dict', 'predict_params': {'test': 1}, 'using': {'test': 1}}
    #     # merge incoming args
    #     # predict_params and having are generally identical, but merged in order to be safe
    #     args = {
    #         **(self.model_storage.json_get('args')),
    #         **dictWithoutNone(args),
    #         **dictWithoutNone(args.get('predict_params',{})),
    #         **dictWithoutNone(args.get('having',{})),
    #     }

    #     print("*** predict merged args", args)
    #     # e.g. {'train_url': 'http://localhost:8000/train', 'predict_url': 'http://localhost:8000/predict', 
    #     # 'dtype_dict': {'timestamp': 'timestamp', 'nodepath': 'string', 'rx_mb_s': 'float', 'date': 'string', 'time_of_day': 'int', 'day_of_week': 'int', 'market_phase': 'int'}, 
    #     # 'format': 'ray_server', 'data_source': 'questdb', 'target': 'rx_mb_s', 'pred_format': 'dict', 
    #     # 'predict_params': {'test': 1}, 
    #     # 'using': {'test': 1}}
    #     resp = requests.post(
    #         args['predict_url'],
    #         json={
    #             'df': df.to_json(orient='records'),
    #             'params': args,
    #         },
    #         headers={'content-type': 'application/json; format=pandas-records'}
    #     )
    #     response = resp.json()
        
    #     target = args['target']
    #     if target != 'prediction':
    #         # rename prediction to target
    #         response[target] = response.pop('prediction')

    #     if response.get('error', None) or resp.status_code > 299 :
    #       print("error returned by ray ML handler", e)
    #       raise Exception("Error - test is: " + response.get('error', 'no message') + " . Response JSON " + str(response) )
        
    #     predictions = pd.DataFrame(response)
        
    #     return predictions
    
    def predict(self, df, args=None):
        args = {**(self.model_storage.json_get('args')), **args}  # merge incoming args
        pred_args = args.get('predict_params', {})
        args = {**args, **pred_args}  # merge pred_args
        if args.get('is_parquet', False):
            buffer = io.BytesIO()
            df.attrs['pred_args'] = pred_args
            df.to_parquet(buffer)
            resp = requests.post(args['predict_url'],
                                 files={"df": ("df", buffer.getvalue(), "application/octet-stream")},
                                 data={"pred_args": json.dumps(pred_args)},
                                 )
        else:
            resp = requests.post(args['predict_url'],
                                 json={'df': df.to_json(orient='records'), 'pred_args': pred_args},
                                 headers={'content-type': 'application/json; format=pandas-records'})
        try:
            if args.get('is_parquet', False):
                buffer = io.BytesIO(resp.content)
                table = pq.read_table(buffer)
                response = table.to_pandas()
            else:
                response = resp.json()
        except json.JSONDecodeError:
            error = resp.text
        except Exception:
            error = 'Could not decode parquet.'
        else:
            if 'prediction' in response:
                target = args['target']
                if target != 'prediction':
                    # rename prediction to target
                    response[target] = response.pop('prediction')
                return pd.DataFrame(response)
            else:
                # something wrong
                error = response

        raise RayServeException(f"Error: {error}")

    def describe(self, key: Optional[str] = None) -> pd.DataFrame:
        args = self.model_storage.json_get('args')
        description = {
            'TRAIN_URL': [args['train_url']],
            'PREDICT_URL': [args['predict_url']],
            'TARGET': [args['target']],
        }
        return pd.DataFrame.from_dict(description)
