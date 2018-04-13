#!/bin/bash

python download.py 150ChYdr4HMukrdoJuROdrzZxiLRFEJov data.zip
unzip data.zip
mkdir -p logs
rm -rf data.zip
