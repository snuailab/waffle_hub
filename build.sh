pip install -U build
python -m build

version=$(python -c "import waffle_hub; print(waffle_hub.__version__)")
docker build . -t snuailab/waffle -t snuailab/waffle:$version --file docker/Dockerfile