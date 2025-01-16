sudo apt-get install -y squashfs-tools
docker save -o dgx-docker.tar argnctu/dgx:gpu
mkdir -p extracted_image
tar -xvf dgx-docker.tar -C extracted_image
mksquashfs extracted_image dgx-docker.sqsh -comp xz
unsquashfs -ll dgx-docker.sqsh
