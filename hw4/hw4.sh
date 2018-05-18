#!/bin/bash
wget -O 'ACGAN_D_e20.pt' 'https://www.dropbox.com/s/ijs5f7cgqitxcsm/ACGAN_D_e20_.pt?dl=1'
wget -O 'ACGAN_G_e20.pt' 'https://www.dropbox.com/s/30c19nd846aj8zr/ACGAN_G_e20_.pt?dl=1'
wget -O 'GAN_D_e25.pt' 'https://www.dropbox.com/s/vz3belji3a0bmwd/GAN_D_e25_.pt?dl=1'
wget -O 'GAN_G_e25.pt' 'https://www.dropbox.com/s/041v2tb8ysxudnd/GAN_G_e25_.pt?dl=1'
wget -O 'VAE_e100_lKL1e-5.pt' 'https://www.dropbox.com/s/pardbcxuyqf8ma7/VAE_e100_lKL1e-5.pt?dl=1'
mkdir ./log
cd ./log
wget -O 'log.zip' 'https://www.dropbox.com/sh/m1dbkjnl2uim85v/AADkH4PKwLuksC-m-EKRe2hoa?dl=1'
unzip log.zip
cd ..
python3 VAE_test.py VAE_e100_lKL1e-5.pt $1 $2
python3 GAN_test.py GAN_G_e25.pt $2
python3 ACGAN_test.py ACGAN_G_e20.pt $2
python3 read_tb_event.py ./log $2

