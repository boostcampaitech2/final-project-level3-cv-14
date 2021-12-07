# Image-Inpainting Environment Setting
- https://github.com/saic-mdal/lama#:~:text=by%20%40Moldoteck%2C%20code-,Environment%20setup,-Clone%20the%20repo
- `pip install -U people_segmentation`

# pretrained model
- [download model](https://drive.google.com/drive/folders/1U7PsxDzC1CnYdNrQP9CkELxF_BcRXEIp?usp=sharing)
- big-lama.tar.gz파일을 `Image-Inpainting/lama/LaMa_models` 아래 두고 `tar -zxvf big-lama.tar.gz` 명령어로 압축 해제

# 실행
- 기본 Image-Inpainting : `streamlit run app.py --server.port 6006`
- People Segmentation 기능 : `streamlit run people_segmentation_app.py --server.port 6006`

# TODO
- People Segmentation 기능을 모듈화하여 app.py에 추가 (~12.08 피어세션 전까지)
