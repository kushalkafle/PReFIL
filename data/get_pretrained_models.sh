#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MMVtvhFCgDNdF4RalzRwlGMQ4D5qNqVE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MMVtvhFCgDNdF4RalzRwlGMQ4D5qNqVE" -O pretrained.tar.gz && rm -rf /tmp/cookies.txt
tar -xvzf pretrained.tar.gz
rm pretrained.tar.gz