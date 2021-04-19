FROM alpine:3.13

RUN apk add wireshark tshark &&\
    chgrp wireshark /usr/bin/dumpcap &&\
    apk add --update --no-cache python3 &&\
    apk add --update py3-pip

RUN pip3 install multiprocess