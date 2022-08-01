
Prerequisite
```sh
$ python3 -m venv venv
$ . venv/bin/activate
(venv) $ pip install -U pip
(venv) $ pip install -r requirements.txt
(venv) $ apt install ffmpeg # For convert dataset.
(venv) $ patch -p1 -d venv < asr_japanese_hg.patch
```
Model Download/Conversion
```sh
(venv) $ omz_downloader --name wav2vec2-large-xlsr-53-japanese
(venv) $ omz_converter --name wav2vec2-large-xlsr-53-japanese
```

Demo
```sh
(venv) $ python infer-file-openvino.py
```

Accuracy Checker
```sh
(venv) $ python3 convert-common-voice-ja.py --dst_dir dataset --split test # download/convert dataset
(venv) $ ./run_accuracy_checker.sh # run accuracy checker 
(venv) $ ACCURACY_CHECKER_LOG_LEVEL=DEBUG ./run_accuracy_checker.sh # print debug message
```

INT8 Quantization
```sh
(venv) $ pot -c default_quantization_template.json
```

```sh
diff -u -r -N venv_orig/lib/python3.8/site-packages/openvino/ venv/lib/python3.8/site-packages/openvino | sed '/Binary\ files\ /d' > asr_japanese_hg.patch
```

