#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ" -O images.tar.gz && rm -rf /tmp/cookies.txt
mkdir -p DVQA/images/train/;
tar -xzvf images.tar.gz -C DVQA/images/train/ --strip-components=1
rm images.tar.gz
cd DVQA/images/; ln -s train/ val_easy; ln -s train/ val_hard
cd ../..
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Tu9FWR1qawNLhVZh9QE4YEDiBVOWEWtn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Tu9FWR1qawNLhVZh9QE4YEDiBVOWEWtn" -O DVQA_qa.tar.gz && rm -rf /tmp/cookies.txt
mkdir -p DVQA/qa/
tar -xzvf DVQA_qa.tar.gz -C DVQA/qa/
rm DVQA_qa.tar.gz