#!/bin/bash
echo -e "\e[31mPlease Download all the files from the FigureQA website prior to running this\e[0m"
mkdir -p FigureQA/images/train; mkdir FigureQA/images/val1; mkdir FigureQA/images/val2; mkdir FigureQA/images/test1/; mkdir FigureQA/images/test2/
tar -xzvf figureqa-train1-v1.tar.gz -C FigureQA/images/train/ --strip-components=2
tar -xzvf figureqa-validation1-v1.tar.gz -C FigureQA/images/val1/ --strip-components=2
tar -xzvf figureqa-validation2-v1.tar.gz -C FigureQA/images/val2/ --strip-components=2
tar -xzvf figureqa-test1-v1.tar.gz -C FigureQA/images/test1/ --strip-components=2
tar -xzvf figureqa-test2-v1.tar.gz -C FigureQA/images/test2/ --strip-components=2
rm figureqa-*.tar.gz
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1B0FJpj-0AYRaxhAlE8tRbX148NBjK7xQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1B0FJpj-0AYRaxhAlE8tRbX148NBjK7xQ" -O FigureQA_qa.tar.gz && rm -rf /tmp/cookies.txt
mkdir -p FigureQA/qa/; tar -xzvf FigureQA_qa.tar.gz -C FigureQA/qa/
rm FigureQA_qa.tar.gz