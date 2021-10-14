# FastSpeech2 Pipeline
![CI](https://github.com/lunarbridge/fastspeech2-pipeline-component/actions/workflows/ci.yml/badge.svg)
[![Component v2 CI/CD](https://github.com/lunarbridge/fastspeech2-pipeline/actions/workflows/ci_v2.yml/badge.svg)](https://github.com/lunarbridge/fastspeech2-pipeline/actions/workflows/ci_v2.yml)

This repository does dockerize modules in [FastSpeech2](https://github.com/AppleHolic/FastSpeech2)  and push them to Docker Hub.
The dockerized components would be used at MLOps pipeline that build on Kubeflow.
## Strucutres
* workflow: GitHub Action that does CI to the components
* components: Components source
## Setup Environment
If you cloned this repository, you should set Docker Hub access token and set secret values in your repository secrets to run CI workflow.
Instructions are described at https://docs.docker.com/ci-cd/github-actions/

# References
* Original codes: [AppleHolic/FastSpeech2](https://github.com/AppleHolic/FastSpeech2)
