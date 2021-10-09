# Usage
```
docker run -it \
       -v /local-storage:/mnt \
       $DOCKER_HUB_USERNAME/fs2-mfa-aligner \
       --mfa-input-path /mnt/data/fastspeech2-2/vctk \
       --lexicon-path /mnt/data/lexicons/librispeech-lexicon.txt \
       --mfa-output-path /mnt/data/fastspeech2-2/vctk-pre
```
