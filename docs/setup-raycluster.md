# Setup: Ray on Kubernetes

This guide covers deploying LLMZero with the **Ray cluster** compute backend on Kubernetes (e.g. Amazon EKS). In this mode, LLMZero runs inside a Kubernetes pod and submits training jobs to a remote Ray cluster via the Ray Job Submission API.

## Prerequisites

1. **kubectl**: [Installation guide](https://kubernetes.io/docs/tasks/tools/)
2. **jq**: `brew install jq` (macOS) or `apt-get install jq` (Linux)
3. **Cluster access**: Authenticate to your Kubernetes cluster (e.g. `aws eks update-kubeconfig`) and set your default namespace.
4. **A running Ray cluster** with your training framework (verl) installed in the worker image. The [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) is the standard way to deploy one.

## Deploy a Ray Cluster

Deploy a Ray cluster with the [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html). Your worker container image must include:

- Ray (matching the head/worker versions)
- Your training framework: `pip install "verl==0.9.0"` (see [setup-conda.md](setup-conda.md)), or build the GPU image in [docker/](../docker/)
- CUDA-enabled PyTorch matching your GPU nodes

A minimal `RayCluster` custom resource defines a head group and one or more GPU worker groups. See the [KubeRay RayCluster docs](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/raycluster-quick-start.html) for a complete template.

Get the Ray head service name for job submission:

```bash
kubectl get svc | grep head-svc
```

## Connect LLMZero to the Ray Cluster

Run LLMZero from a pod (or any host) with network access to the Ray head service, specifying the service via CLI flags:

```bash
llmzero \
    -i /path/to/data \
    -t "Your instruction" \
    --ray-service my-ray-cluster-head-svc \
    --ray-namespace llmzero \
    --ray-port 8265
```

Or set defaults in the config file (`src/autogluon/llmzero/configs/default.yaml`):

```yaml
compute_infra:
  type: raycluster
  kubernetes_namespace: "llmzero"
  ray_service_name: "my-ray-cluster-head-svc"
  ray_dashboard_port: 8265
```

## How It Works

The raycluster backend:
1. Wraps the generated training script in a `ray job submit` call targeting the Ray dashboard
   (`http://<ray-service>.<namespace>.svc.cluster.local:<port>`)
2. Streams the remote job's stdout line-by-line so observers can parse metrics and validation
   scores in real time
3. Tracks the Ray job ID for status polling and termination

To stop a running job:

```bash
llmzero stop-job <job-id> --compute-backend raycluster --ray-service my-ray-cluster-head-svc --ray-namespace llmzero
```
