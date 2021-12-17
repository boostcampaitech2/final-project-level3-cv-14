import gdown

google_path='https://drive.google.com/uc?id='
file_id = '1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb'
output_name = "model_deblurring.pth"
gdown.download(google_path+file_id,output_name,quiet=False)