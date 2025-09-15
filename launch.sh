#!/usr/bin/env bash
# launch_eu_gpu.sh  –  valid as of 2025-09-15

set -euo pipefail

KEY=id_rsa_default
VOL_SIZE=10000          # 10 TB root volume
SG_ID=sg-0add4c7b3d712d7d5
REGIONS=(eu-west-1 eu-central-1 eu-west-2 eu-north-1)

# Ordered by single-GPU VRAM (high → low)
INSTANCE_TYPES=(
  # H200
  p5en.48xlarge  # 8×H200 141 GB each
  p5e.48xlarge   # 8×H200 141 GB each
  # H100
  p5.4xlarge     # 1×H100 80 GB
  p5.48xlarge    # 8×H100 80 GB each
  # L40S
  g6e.xlarge g6e.2xlarge g6e.4xlarge g6e.8xlarge g6e.12xlarge g6e.24xlarge
  # A100
  p4d.24xlarge p4de.24xlarge
  # A10G
  g5.xlarge g5.2xlarge g5.4xlarge g5.8xlarge g5.12xlarge
  # L4
  g6.xlarge g6.2xlarge g6.4xlarge g6.8xlarge g6.12xlarge
  # T4
  g4dn.xlarge g4dn.2xlarge g4dn.4xlarge g4dn.8xlarge g4dn.12xlarge
)

gpu_info() {
  case $1 in
    p5en.*|p5e.*) echo "H200 (141 GB)";;
    p5.*)         echo "H100 (80 GB)";;
    g6e.*)        echo "L40S (48 GB)";;
    p4d.*|p4de.*) echo "A100 (40 GB)";;
    g5.*)         echo "A10G (24 GB)";;
    g6.*)         echo "L4 (24 GB)";;
    g4dn.*)       echo "T4 (16 GB)";;
    *)            echo "Unknown";;
  esac
}

echo "🇪🇺 Launching in EU regions (valid types only)..."

for region in "${REGIONS[@]}"; do
  echo "Importing key to $region..."
  aws ec2 import-key-pair --region "$region" --key-name "$KEY" \
       --public-key-material fileb://~/.ssh/id_ed25519.pub 2>/dev/null \
       || echo "Key already exists in $region"

  for type in "${INSTANCE_TYPES[@]}"; do
    GPU_INFO=$(gpu_info "$type")
    echo "🔄 Trying $type ($GPU_INFO) in $region..."

    AMI_ID=$(aws ssm get-parameter --region "$region" \
              --name /aws/service/deeplearning/ami/x86_64/multi-framework-oss-nvidia-driver-amazon-linux-2/latest/ami-id \
              --query 'Parameter.Value' --output text 2>/dev/null) || true
    [[ -z "$AMI_ID" ]] && { echo "❌ No DLAMI in $region"; continue; }

    if aws ec2 run-instances \
         --region "$region" \
         --instance-type "$type" \
         --image-id "$AMI_ID" \
         --key-name "$KEY" \
         --security-group-ids "$SG_ID" \
         --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOL_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
         --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-training}]' >/dev/null; then
      echo "✅ Launched $type ($GPU_INFO) in $region"
      sleep 15
      IP=$(aws ec2 describe-instances --region "$region" \
           --filters "Name=tag:Name,Values=ai-training" \
                     "Name=instance-state-name,Values=running,pending" \
           --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
      echo -e "\n🚀 $type in $region – ssh ec2-user@$IP\n"
      exit 0
    else
      echo "❌ Failed $type in $region"
    fi
  done
done

echo "❌ No capacity or supported types available in EU regions"
exit 1