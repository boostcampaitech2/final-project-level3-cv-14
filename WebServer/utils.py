import toml
from oauth2client.service_account import ServiceAccountCredentials
from google.cloud import storage

def send_to_bucket(image_name, image_bytes):
    """ 
    gcp storage에 올라가는 public url을 만든다.
    """
    secrets = toml.load("WebServer/secret/secrets.toml")
    storage_client = storage.Client.from_service_account_json(
        'WebServer/secret/secrets_gcp.json')

    bucket = storage_client.get_bucket(secrets['gcp']['bucket'])
    bucket.blob(image_name).upload_from_string(image_bytes)
    image_url = bucket.blob(image_name).public_url

    return image_url