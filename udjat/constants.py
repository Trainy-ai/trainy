# Default Ray port is 6379. Default Ray dashboard port is 8265.
# Default SkyPilot port is 6380. Default Skypilot dashboard 8266
# Default Ray tempdir is /tmp/ray.
# We change them to avoid conflicts with user's Ray clusters.
# We note down the ports in ~/.sky/ray_port.json for backward compatibility.
UDJAT_REMOTE_RAY_PORT = 6381
UDJAT_REMOTE_RAY_DASHBOARD_PORT = 8261
UDJAT_REMOTE_RAY_CLIENT_PORT = 10001
UDJAT_TMPDIR = "/tmp/ray_udjat"
