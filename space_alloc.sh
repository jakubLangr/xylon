# Format the 10TB disk
sudo mkfs.ext4 /dev/nvme1n1

# Create mount point
sudo mkdir -p /data

# Mount the disk
sudo mount /dev/nvme1n1 /data

# Make it permanent (survives reboots)
echo '/dev/nvme1n1 /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab

# Change ownership to ec2-user
sudo chown ec2-user:ec2-user /data

# Verify
df -h /data

sudo yum install -y htop

sudo fallocate -l 32G /data/swapfile
sudo chmod 600 /data/swapfile
sudo mkswap /data/swapfile
sudo swapon /data/swapfile