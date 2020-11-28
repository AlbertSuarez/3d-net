import time
import requests

from src.helper import log


def execute(method, url, params=None):
    if method not in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE'):
        log.error('[ERROR] Indicated method must be GET, POST, PUT, PATCH or DELETE')
        return None
    for it in range(1, 4):
        try:
            response = requests.request(method=method, url=url, params=params)
            if response is not None and response.ok:
                return response.json()
            else:
                raise Exception(f'Problem requesting URL {url}. Number of tries: {it}')
        except Exception as e:
            log.exception(e)
            time.sleep(it)
    return None
