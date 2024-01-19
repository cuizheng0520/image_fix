## Batch Watermarks Remover

Remove text watermarks in batches.
用于从图片中批量自动移除文字水印。它使用 Paddle ocr、OpenCV 实现水印识别与移除

install:

```markdown
pip install -r requirements.txt 
```

run demo:

```markdown
python demo.py
```

流程:

- 图像读取：OpenCV
- 文字识别：PaddleHub 的 chinese_ocr_db_crnn_server 模型
- 水印定位：(可选) 水印区域扩展微调
- 水印移除：OpenCV 水印移除
- 过程调试：Matplotlib
