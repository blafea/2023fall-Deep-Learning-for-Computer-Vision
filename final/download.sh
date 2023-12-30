#!bin/bash
mkdir model_final
gdown -O model_final/best_model.pth.tar 1owmTgeOMv16FiueEvtyVIO-OwM6hofUU

mkdir student_data
gdown -O student_data/image_hw.zip 126XeUBKzLymvrT67cC0zhH-Hw8nYb1cF
cd student_data
unzip image_hw.zip
rm image_hw.zip