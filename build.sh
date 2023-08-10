pip install -U build
python -m build

bash download.sh
version=$(python -c "import waffle_hub; print(waffle_hub.__version__)")
wd sample --name mnist_det --task object_detection
wd sample --name mnist_cls --task classification
docker build . --build-arg NO_CACHE_ARG=$(date +%s) -t snuailab/waffle -t snuailab/waffle:$version --file docker/Dockerfile