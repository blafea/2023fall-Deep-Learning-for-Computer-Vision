gdown "https://drive.google.com/u/3/uc?id=1oW4FLcqsVt9RltRgWZTKY0txdxtBGTd5&export=download" -O "best_model.pth"

python3 -c "import clip; clip.load('ViT-L/14')"
python3 -c "import clip; clip.load('ViT-B/32')"
python3 -c "import timm; timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True)"