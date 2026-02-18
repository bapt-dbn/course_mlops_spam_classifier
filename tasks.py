import os
import signal
import time

import httpx
from invoke.context import Context
from invoke.tasks import task

from course_mlops.utils import EnvironmentVariable

PTY = not os.getenv("CI")

MAX_RETRIES = 30
RETRY_INTERVAL = 5

CHART_PATH = "chart"
K8S_RELEASE = "course-mlops"
K8S_NAMESPACE = "course-mlops"


def _wait_for_service(c: Context, url: str, name: str) -> None:
    for i in range(1, MAX_RETRIES + 1):
        try:
            response = httpx.get(url, timeout=5)
            if response.status_code == 200:
                print(f"{name} is ready.")
                return
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError):
            pass
        print(f"Waiting for {name}... ({i}/{MAX_RETRIES})")
        time.sleep(RETRY_INTERVAL)
    c.run("docker compose logs", pty=PTY)
    raise RuntimeError(f"{name} failed to start after {MAX_RETRIES * RETRY_INTERVAL}s")


@task
def compose_up(c: Context, build: bool = False, attach: bool = False, no_migrate: bool = False) -> None:
    if build:
        c.run("docker compose build --quiet", pty=PTY)

    c.run("docker compose up --quiet-pull --wait", pty=PTY)

    if not no_migrate:
        migrate(c)

    print("Stack started:")
    print("  API:        http://localhost:8000")
    print("  API docs:   http://localhost:8000/docs")
    print("  MLflow UI:  http://localhost:5001")
    print("  MinIO UI:   http://localhost:9001 (minio/password)")

    if attach:
        c.run("docker compose logs --follow", pty=True)


@task
def compose_down(c: Context, volumes: bool = False) -> None:
    flag = "-v" if volumes else ""
    c.run(f"docker compose down {flag}", pty=PTY)
    print("Docker Compose stopped.")


@task
def test(c: Context, coverage: bool = False, min_coverage: int = 95) -> None:  # noqa: PT028
    """Run unit tests with optional coverage."""
    coverage_opts = (
        f"--cov=course_mlops/train --cov-report=term-missing --cov-fail-under={min_coverage}" if coverage else ""
    )
    c.run(f"uv run pytest tests/unit {coverage_opts}", pty=PTY)


@task(name="integration-test")
def integration_test(c: Context) -> None:
    """Start the stack, train a model, run integration tests, then tear down."""
    try:
        print("Building API image...")
        c.run("docker compose build --quiet api", pty=PTY)

        print("Starting infrastructure (Postgres + MinIO + MLflow)...")
        c.run("docker compose up -d --quiet-pull mlflow", pty=PTY)
        _wait_for_service(c, "http://localhost:5001/health", "MLflow")

        print("Applying migrations...")
        migrate(c)

        print("Training model...")
        c.run(
            "uv run mlops_course train",
            env={
                "MLFLOW_TRACKING_URI": "http://localhost:5001",
                "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio-secret",
            },
            pty=PTY,
        )

        print("Starting API...")
        c.run("docker compose up -d api", pty=PTY)
        _wait_for_service(c, "http://localhost:8000/api/v1/health", "API")

        print("Running integration tests...")
        c.run("uv run pytest tests/integration -m integration", pty=PTY)
    finally:
        print("Tearing down stack...")
        c.run("docker compose down -v", pty=PTY)


@task
def migrate(c: Context) -> None:
    """Apply database migrations (Alembic)."""
    c.run(
        "uv run mlops_course migrate",
        env={
            EnvironmentVariable.DB_HOST: "localhost",
            EnvironmentVariable.DB_PORT: "5432",
            EnvironmentVariable.DB_USER: "postgres",
            EnvironmentVariable.DB_PASSWORD: "password",
            EnvironmentVariable.DB_NAME: "postgres",
        },
        pty=PTY,
    )


@task
def train(c: Context, all_models: bool = False) -> None:
    cmd = "uv run mlops_course train"
    if all_models:
        cmd += " --all"

    c.run(
        cmd,
        env={
            "MLFLOW_TRACKING_URI": "http://localhost:5001",
            "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio-secret",
        },
        pty=PTY,
    )

    c.run("docker compose restart api", pty=PTY)
    print("Model trained and API restarted.")


def _minikube_is_running(c: Context) -> bool:
    result = c.run("minikube status --format='{{.Host}}'", hide=True, warn=True)
    return result.ok and "Running" in result.stdout


@task(name="k8s-up")
def k8s_up(c: Context, build: bool = False) -> None:
    """Start minikube, install CNPG operator, and deploy the Helm chart."""
    # 1. Start minikube if not running
    if not _minikube_is_running(c):
        print("Starting minikube...")
        c.run("minikube start --cpus=4 --memory=4096", pty=PTY)
    else:
        print("Minikube already running.")

    # 2. Build and load the API image into minikube
    if build:
        print("Building API image...")
        c.run("docker build -t course-mlops-api:local .", pty=PTY)
        c.run("minikube image load course-mlops-api:local", pty=PTY)

    # 3. Create namespace
    c.run(f"kubectl create namespace {K8S_NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -", pty=PTY)

    # 4. Install CloudNativePG operator (idempotent)
    print("Installing CloudNativePG operator...")
    c.run(
        "kubectl apply --server-side -f "
        "https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.28/releases/cnpg-1.28.1.yaml",
        pty=PTY,
    )
    print("Waiting for CNPG operator to be ready...")
    c.run(
        "kubectl rollout status deployment/cnpg-controller-manager -n cnpg-system --timeout=120s",
        pty=PTY,
    )

    # 5. Update Helm dependencies and install/upgrade chart
    print("Deploying Helm chart...")
    c.run(f"helm dependency update {CHART_PATH}", pty=PTY)
    c.run(
        f"helm upgrade --install {K8S_RELEASE} {CHART_PATH} "
        f"--namespace {K8S_NAMESPACE} "
        f"--set api.image.repository=course-mlops-api "
        f"--set api.image.tag=local "
        f"--set api.image.pullPolicy=Never "
        f"--wait --timeout 5m",
        pty=PTY,
    )

    print()
    print("Stack deployed! To access services:")
    print(f"  API:       kubectl port-forward -n {K8S_NAMESPACE} svc/{K8S_RELEASE}-api 8000:8000")
    print(f"  MLflow:    kubectl port-forward -n {K8S_NAMESPACE} svc/{K8S_RELEASE}-mlflow 5001:5001")
    print(f"  MinIO:     kubectl port-forward -n {K8S_NAMESPACE} svc/{K8S_RELEASE}-minio-console 9001:9001")


@task(name="k8s-down")
def k8s_down(c: Context, full: bool = False) -> None:
    """Uninstall the Helm release. Use --full to also stop minikube."""
    c.run(f"helm uninstall {K8S_RELEASE} --namespace {K8S_NAMESPACE}", warn=True, pty=PTY)
    c.run(f"kubectl delete namespace {K8S_NAMESPACE}", warn=True, pty=PTY)
    print("Helm release uninstalled.")

    if full:
        c.run("minikube stop", pty=PTY)
        print("Minikube stopped.")


K8S_PORT_FORWARDS = {
    "api": (f"svc/{K8S_RELEASE}-api", "8000:8000"),
    "mlflow": (f"svc/{K8S_RELEASE}-mlflow", "5001:5001"),
    "minio-api": (f"svc/{K8S_RELEASE}-minio", "9000:9000"),
    "minio-console": (f"svc/{K8S_RELEASE}-minio-console", "9001:9001"),
}


@task(name="k8s-proxy")
def k8s_proxy(c: Context) -> None:
    """Open port-forwards to all UIs (API docs, MLflow, MinIO console). Ctrl+C to stop."""
    import subprocess  # noqa: PLC0415

    processes: list[subprocess.Popen[bytes]] = []

    for name, (svc, ports) in K8S_PORT_FORWARDS.items():
        cmd = ["kubectl", "port-forward", "-n", K8S_NAMESPACE, svc, ports]
        processes.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        print(f"  {name} port-forward started")

    print()
    print("UIs available at:")
    print("  API docs:      http://localhost:8000/docs")
    print("  MLflow UI:     http://localhost:5001")
    print("  MinIO Console: http://localhost:9001  (minio / minio-secret)")
    print()
    print("Press Ctrl+C to stop all port-forwards.")

    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.terminate()
        print("\nPort-forwards stopped.")
