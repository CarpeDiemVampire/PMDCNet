# PMDCNet

## get ready

**python 3.8**;  
**PyTorch 1.8.0**;   
**Numpy >=1.18**;   
**CUDA 11.0**;  
**GCC >=9.0**;   
**pyyaml**;
**cv2**;
**Pillow**;
**imgaug**;

## Datasets
### Download the TextZoom dataset:

	https://github.com/JasonBoy1/TextZoom

## Train
### Download the pretrained recognizer from: 
	Aster: https://github.com/ayumiymk/aster.pytorch  
	MORAN:  https://github.com/Canjie-Luo/MORAN_v2  
	CRNN: https://github.com/meijieru/crnn.pytorch

### Train model: 
 ```
python3 main.py --arch="pmdc" --batch_size=64 --STN --mask --use_distill --gradient  --sr_share --stu_iter=1 --vis_dir='pmdc/'  --rotate_train=5 --rotate_test=0 --learning_rate=0.001 --tssim_loss --test_model="CRNN" 
```


## Eval
### Run the test-prefixed shell to test the corresponding model.
```
Adding '--go_test' in the shell file
```
