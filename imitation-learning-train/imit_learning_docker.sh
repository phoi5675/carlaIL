sudo docker run -it --rm --name imit_learn \
    -v "$(pwd)/imitation:/imitation" \
    -v "$DOCKER_BASH_HISTORY:/root/.bash_history" \
    imit-learn bash
