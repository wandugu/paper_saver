from .experiment import run


def main(ctx, config):
    run(ctx, config, pilot=True)
