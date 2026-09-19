import os


ALPACA_ACCOUNT_ENV_VARS = {
    "alpha_1": (
        "ALPACA_ALPHA_1_API_KEY",
        "ALPACA_ALPHA_1_SECRET_KEY",
    ),
    "alpha_2": (
        "ALPACA_ALPHA_2_API_KEY",
        "ALPACA_ALPHA_2_SECRET_KEY",
    ),
}


def load_alpaca_credentials(account, environ=None):
    try:
        api_key_variable, secret_key_variable = ALPACA_ACCOUNT_ENV_VARS[account]
    except KeyError as error:
        raise ValueError(f"Unsupported Alpaca account: {account}") from error

    environment = os.environ if environ is None else environ
    api_key = environment.get(api_key_variable)
    secret_key = environment.get(secret_key_variable)

    if not api_key or not secret_key:
        raise ValueError(
            f"Missing Alpaca credentials for account: {account}"
        )

    return api_key, secret_key
