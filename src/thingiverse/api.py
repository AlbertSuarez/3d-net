import os
import flask
import dotenv
import requests

from urllib.parse import quote, parse_qsl
from flask_cors import CORS

from src.config import THINGIVERSE_API_ENDPOINT_AUTH, THINGIVERSE_API_ENDPOINT_TOKEN, THINGIVERSE_API_DONE, \
    THINGIVERSE_FLASK_ENDPOINT
from src.helper import log
from src.thingiverse import thing


dotenv.load_dotenv('.env')
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.secret_key = os.environ.get('THINGIVERSE_API_CLIENT_SECRET')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
CORS(app)


@app.route(THINGIVERSE_FLASK_ENDPOINT, methods=['GET'])
def latest_things():
    try:
        if 'credentials' not in flask.session:
            return flask.redirect(flask.url_for(oauth2_authorize.__name__))
        thing.get_latest(flask.session.get('credentials', {}).get('access_token'))
        return flask.redirect(THINGIVERSE_API_DONE)
    except Exception as e:
        message = f'Exception in {latest_things.__name__} function: [{e}]'
        log.error(message)
        log.exception(e)
        return flask.jsonify(dict(error=True, response=message))


@app.route('/oauth2/authorize', methods=['GET'])
def oauth2_authorize():
    try:
        auth_query_parameters = dict(
            client_id=os.environ.get('THINGIVERSE_API_CLIENT_ID'),
            redirect_uri=flask.url_for(oauth2_callback.__name__, _external=True),
            response_type='code'
        )
        url_args = '&'.join(['{}={}'.format(key, quote(val)) for key, val in auth_query_parameters.items()])
        auth_url = '{}/?{}'.format(THINGIVERSE_API_ENDPOINT_AUTH, url_args)
        return flask.redirect(auth_url)
    except Exception as e:
        message = f'Exception in {oauth2_authorize.__name__} function: [{e}]'
        log.error(message)
        log.exception(e)
        return flask.jsonify(dict(error=True, response=message))


@app.route('/oauth2/callback', methods=['GET'])
def oauth2_callback():
    try:
        auth_code = flask.request.args.get('code')
        auth_query_parameters = dict(
            client_id=os.environ.get('THINGIVERSE_API_CLIENT_ID'),
            client_secret=os.environ.get('THINGIVERSE_API_CLIENT_SECRET'),
            code=auth_code
        )
        response = requests.post(THINGIVERSE_API_ENDPOINT_TOKEN, params=auth_query_parameters)
        query_string = dict(parse_qsl(response.text))
        flask.session['credentials'] = dict(access_token=query_string.get('access_token'))
        return flask.redirect(flask.url_for(latest_things.__name__))
    except Exception as e:
        message = f'Exception in {oauth2_callback.__name__} function: [{e}]'
        log.error(message)
        log.exception(e)
        return flask.jsonify(dict(error=True, response=message))
