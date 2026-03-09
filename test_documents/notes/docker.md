# Docker Quick Notes

Docker is a containerization platform that allows developers to package applications and dependencies into portable containers.

## Installing Docker on Ubuntu

Run the following commands:

sudo apt update
sudo apt install docker.io

After installation, start Docker:

sudo systemctl start docker

## Running a Container

Example command:

docker run hello-world