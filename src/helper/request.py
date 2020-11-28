import time
import requests

from src.config import REQUEST_TIMEOUT
from src.helper import log


def execute(method, url, params=None):
    if method not in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE'):
        log.error('[ERROR] Indicated method must be GET, POST, PUT, PATCH or DELETE')
        return None
    for it in range(1, 4):
        try:
            response = requests.request(method=method, url=url, params=params, timeout=REQUEST_TIMEOUT)
            if response is not None and response.ok:
                return response.json()
            else:
                raise Exception(
                    f'Problem requesting URL {url}: '
                    f'[{response.status_code if response is not None else None}]. Number of tries: {it}'
                )
        except Exception as e:
            time.sleep(it)
            if it >= 3:
                log.debug(f'Error: {e}')
    return None
