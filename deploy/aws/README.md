# Deploying to AWS App Runner

App Runner runs a container behind HTTPS with autoscaling and health checks, without a cluster to manage. It suits a small, stateless API like this one. The steps below are written to be followed once. I haven't run them, because they need an AWS account, so expect to adjust names to your setup.

## One-time setup

1. **Create the ECR repository** for the image:
   ```bash
   aws ecr create-repository --repository-name jetway --region us-east-2
   ```
2. **Let GitHub deploy without stored keys.** In IAM, add GitHub as an OIDC identity provider (`token.actions.githubusercontent.com`). Then create a role that trusts your repository and can push to ECR and call `apprunner:UpdateService`. Save its ARN as the repository secret `AWS_DEPLOY_ROLE_ARN`.
3. **Push a first image by hand**, so there's something to start the service from:
   ```bash
   aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-2.amazonaws.com
   docker build -t <account>.dkr.ecr.us-east-2.amazonaws.com/jetway:first .
   docker push <account>.dkr.ecr.us-east-2.amazonaws.com/jetway:first
   ```
4. **Create the App Runner service** in the console from that image: port 8000, health check path `/health`, 1 vCPU and 2 GB. Create or pick an access role that lets App Runner pull from ECR. Save the service ARN as `APP_RUNNER_SERVICE_ARN` and the access role ARN as `APP_RUNNER_ECR_ROLE_ARN`.
5. **Optional:** to enable model-written replies on `/draft`, add `ANTHROPIC_API_KEY` as an App Runner environment variable stored in Secrets Manager. Without it, replies use the extractive fallback.

## Each deploy

Run the **deploy-aws** workflow from the Actions tab. It builds the image, tags it with the commit, pushes it to ECR, and points the service at it.

## In production, I'd also

- Send `/metrics` to CloudWatch, or scrape it with Amazon Managed Prometheus, and alarm on handoff rate and drafts that fail the grounding check.
- Write the prediction log to S3 instead of local disk, since App Runner storage doesn't persist. Then run `src/monitor.py` on a daily schedule against it.
- Keep models in S3 with the registry file, so a new model is a config change rather than a new image.
