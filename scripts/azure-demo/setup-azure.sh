#!/usr/bin/env bash
# Run from the repository on your Mac after az login and gh auth login.
# Creates a Basic ACR registry and a GitHub federated managed identity.
# Does not push code or deploy the application.
set -euo pipefail

REPO="Mohamed1-2/aurora"
RG="aurora-rg"
AKS="aurora"
IDENTITY="aurora-github"
for tool in az gh kubectl kubelogin; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool"; exit 1; }
done
az account set --subscription "Azure subscription 1"
SUB=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)
LOCATION=$(az aks show -g "$RG" -n "$AKS" --query location -o tsv)
AKS_ID=$(az aks show -g "$RG" -n "$AKS" --query id -o tsv)
PRIVATE=$(az aks show -g "$RG" -n "$AKS" --query apiServerAccessProfile.enablePrivateCluster -o tsv)
RANGES=$(az aks show -g "$RG" -n "$AKS" --query 'apiServerAccessProfile.authorizedIpRanges' -o tsv)
if [[ "$PRIVATE" == "true" || -n "$RANGES" ]]; then
  echo "This cluster restricts API network access. A GitHub-hosted runner may not reach it."
  echo "Stop here and configure a runner with approved network access; do not remove the restrictions."
  exit 1
fi
gh repo view "$REPO" --json nameWithOwner,defaultBranchRef
BRANCH=$(gh repo view "$REPO" --json defaultBranchRef --jq '.defaultBranchRef.name')
[[ "$BRANCH" == "main" ]] || { echo "This guide expects main; found $BRANCH"; exit 1; }

# Reuse the registry recorded in GitHub on subsequent runs.
# If you already created another registry, run: export AURORA_ACR=itsname
ACR="${AURORA_ACR:-}"
if [[ -z "$ACR" ]]; then
  ACR=$(gh variable get ACR_NAME -R "$REPO" 2>/dev/null || true)
fi
if [[ -z "$ACR" ]]; then
  ACR="aurora$(printf '%s' "$SUB" | tr -d '-' | cut -c1-12)"
fi
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.ManagedIdentity --wait
if ! az acr show -g "$RG" -n "$ACR" >/dev/null 2>&1; then
  az acr create -g "$RG" -n "$ACR" -l "$LOCATION" \
    --sku Basic --admin-enabled false --role-assignment-mode rbac
fi
ACR_MODE=$(az acr show -g "$RG" -n "$ACR" --query roleAssignmentMode -o tsv)
if [[ "$ACR_MODE" == "AbacRepositoryPermissions" ]]; then
  echo "Existing registry uses ABAC. This setup expects ordinary registry RBAC."
  exit 1
fi
ACR_ID=$(az acr show -g "$RG" -n "$ACR" --query id -o tsv)
SERVER=$(az acr show -g "$RG" -n "$ACR" --query loginServer -o tsv)
az aks update -g "$RG" -n "$AKS" --attach-acr "$ACR_ID"

az identity create -g "$RG" -n "$IDENTITY" -l "$LOCATION" >/dev/null
CLIENT=$(az identity show -g "$RG" -n "$IDENTITY" --query clientId -o tsv)
PRINCIPAL=$(az identity show -g "$RG" -n "$IDENTITY" --query principalId -o tsv)
az identity federated-credential create -g "$RG" \
  --identity-name "$IDENTITY" --name github-main \
  --issuer https://token.actions.githubusercontent.com \
  --subject "repo:Mohamed1-2@34875234/aurora@1377914325:ref:refs/heads/main" \
  --audiences api://AzureADTokenExchange >/dev/null

az role assignment create --assignee-object-id "$PRINCIPAL" \
  --assignee-principal-type ServicePrincipal --role AcrPush --scope "$ACR_ID" >/dev/null
az role assignment create --assignee-object-id "$PRINCIPAL" \
  --assignee-principal-type ServicePrincipal \
  --role "Azure Kubernetes Service Cluster User Role" --scope "$AKS_ID" >/dev/null

az aks get-credentials -g "$RG" -n "$AKS" --overwrite-existing
kubelogin convert-kubeconfig -l azurecli
kubectl get nodes
kubectl create namespace aurora --dry-run=client -o yaml | kubectl apply -f -

AZURE_RBAC=$(az aks show -g "$RG" -n "$AKS" --query aadProfile.enableAzureRbac -o tsv)
AAD=$(az aks show -g "$RG" -n "$AKS" --query aadProfile.managed -o tsv)
if [[ "$AZURE_RBAC" == "true" ]]; then
  az role assignment create --assignee-object-id "$PRINCIPAL" \
    --assignee-principal-type ServicePrincipal \
    --role "Azure Kubernetes Service RBAC Writer" \
    --scope "$AKS_ID/namespaces/aurora" >/dev/null
elif [[ "$AAD" == "true" ]]; then
  kubectl create rolebinding aurora-github-edit -n aurora \
    --clusterrole=edit --user="$PRINCIPAL" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "Cluster uses local Kubernetes credentials; its cluster-user credentials may have broad access."
  echo "For tighter isolation later, configure Entra authentication and namespace-scoped Azure RBAC."
fi

gh variable set AZURE_CLIENT_ID -R "$REPO" --body "$CLIENT"
gh variable set AZURE_TENANT_ID -R "$REPO" --body "$TENANT"
gh variable set AZURE_SUBSCRIPTION_ID -R "$REPO" --body "$SUB"
gh variable set AZURE_RESOURCE_GROUP -R "$REPO" --body "$RG"
gh variable set AKS_NAME -R "$REPO" --body "$AKS"
gh variable set ACR_NAME -R "$REPO" --body "$ACR"
gh variable set ACR_LOGIN_SERVER -R "$REPO" --body "$SERVER"
echo "Setup complete. Registry: $SERVER"
echo "Allow several minutes for Azure role assignments to propagate before running the workflow."
gh variable list -R "$REPO"
