import multiprocessing
import os
import time
import webbrowser

from src.config import THINGIVERSE_FLASK_PORT, THINGIVERSE_FLASK_WAIT_PRE, THINGIVERSE_FLASK_ENDPOINT, \
    THINGIVERSE_FLASK_WAIT_POST, THINGIVERSE_FLASK_WAIT_ENABLE
from src.helper import log
from src.thingiverse import launcher


def main():
    if not os.path.isfile('.env'):
        log.error('Environment file (.env) could not be found.')
        return

    log.info(f'Spinning up local API for extracting Thingiverse Thing IDs at port {THINGIVERSE_FLASK_PORT}.')
    process = multiprocessing.Process(target=launcher.run_app, args=())
    process.start()
    log.info(f'Sleeping {THINGIVERSE_FLASK_WAIT_PRE} seconds for letting the API to initialize and open a browser.')
    time.sleep(THINGIVERSE_FLASK_WAIT_PRE)
    webbrowser.open(f'http://localhost:{THINGIVERSE_FLASK_PORT}{THINGIVERSE_FLASK_ENDPOINT}', new=2)
    if THINGIVERSE_FLASK_WAIT_ENABLE:
        log.info(f'Sleeping {THINGIVERSE_FLASK_WAIT_POST} seconds for letting the process to finish.')
        log.info('You can stop the script if the process is done.')
        time.sleep(THINGIVERSE_FLASK_WAIT_POST)
        log.info('Killing processes.')
        process.terminate()
    else:
        log.debug('Processing forever.')


if __name__ == '__main__':
    main()
