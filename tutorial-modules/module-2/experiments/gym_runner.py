"""Own a bounded local Ray cluster and dispatch the native Gym CLI."""

import sys

import click

from experiments.common import progress

RAY_CPUS = 2
OBJECT_STORE_BYTES = 256 * 1024 * 1024
RAY_READY_MESSAGE = "Workshop Ray startup complete."
RAY_STARTUP_TIMEOUT_S = 90
GYM_STARTUP_TIMEOUT_S = 120


def check_worker() -> bool:
    return True


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def main(ctx: click.Context) -> None:
    """Forward Gym arguments after starting and checking the local Ray worker."""
    progress("startup", "Starting local Ray: 2 CPUs, 256 MiB object store, dashboard disabled; deadline 90s.")
    import ray
    from nemo_gym.cli.main import main as gym_main

    try:
        ray.init(num_cpus=RAY_CPUS, object_store_memory=OBJECT_STORE_BYTES, include_dashboard=False)
        if not ray.get(ray.remote(check_worker).remote(), timeout=RAY_STARTUP_TIMEOUT_S):
            raise RuntimeError("Ray worker readiness check failed")
        address = ray.get_runtime_context().gcs_address
        print(RAY_READY_MESSAGE, flush=True)
        progress("startup", "Ray worker ready. Preparing inputs and starting Gym services; deadline 120s.")
        # Gym forwards this supported setting to every component subprocess.
        sys.argv = ["gym", *ctx.args, f"++ray_head_node_address={address}"]
        gym_main()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
