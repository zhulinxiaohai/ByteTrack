from cnocr import CnOcr

img_fp = './tmp/hiv00002/img_000001.jpg'
ocr = CnOcr()  # 所有参数都使用默认值
out = ocr.ocr(img_fp)

print(out)