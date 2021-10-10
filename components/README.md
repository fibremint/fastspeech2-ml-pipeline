# Components
dockerized components
## fs2-mfa-aligner
__Usage example__
```
docker run -it -v /local-storage:/mnt \
    $DOCKER_HUB_USER/fs2-mfa-aligner \
    --current-data-path /mnt/fs2-data/data/20211010-000223-intermediate
```
## fs2-prepare-align
__Usage example__
```
docker run -it -v /local-storage:/mnt \
    $DOCKER_HUB_USER/fs2-prepare-align \
    --data-base-path /mnt/fs2-data/data/20211010-000223-intermediate
```
## fs2-prepare-data
__Usage example__
```
docker run -it -v /local-storage:/mnt \
    $DOCKER_HUB_USER/fs2-prepare-data
```
## fs2-preprocess
__Usage example__
```
docker run -it -v /local-storage:/mnt \
    $DOCKER_HUB_USER/fs2-preprocess \
    --data-base-path /mnt/fs2-data/data/20211010-000223-intermediate
```
