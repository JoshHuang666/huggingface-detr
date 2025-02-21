docker create --name temp_container argnctu/huggingface-detr:ros1-gpu
docker export temp_container > dgx_gpu_flat.tar
mkdir extracted_image
tar -xvf dgx_gpu_flat.tar -C extracted_image
sudo rm dgx_gpu_flat.tar
docker rm temp_container
sudo mksquashfs extracted_image dgx_gpu.sqsh -comp lz4 -Xhc
sudo rm -rf extracted_image