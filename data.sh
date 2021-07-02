#!/bin/bash

gfile=https://drive.google.com/file/d/1cGYHnsm6Frq4lp0pBTwU5xsQdx0wa5T8/view?usp=sharing

gfile_ID=(${gfile//// })
echo ${gfile_ID[-2]}
mkdir data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${gfile_ID[-2]} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${gfile_ID[-2]}" -O data/data.tar.gz && rm -rf /tmp/cookies.txt && tar -xvzf data/data.tar.gz && rm data/data.zip