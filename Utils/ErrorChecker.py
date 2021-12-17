import sentry_sdk
from sentry_sdk import capture_exception

class Sentry():
    def __init__(self):
        sentry_sdk.init(
            "https://6b0230d295764e5ea4480188dc2125a1@o1092735.ingest.sentry.io/6111403",

            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=1.0,
        )

    def check(self,e):
        capture_exception(e)
        raise e