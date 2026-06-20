fs = 16000
'''
如果在訓練時遇到 CUDA Out of Memory，建議你可以：

將 chunk_len 降為 3 或 4。

或者縮小 Batch Size。
chunk_len = 5  # (s)
'''
chunk_len = 5  # (s)
chunk_size = chunk_len * fs
num_spks = 2

# network configure
# nnet_conf = {
#     "L": 16,
#     "N": 512,
#     "X": 8,
#     "R": 1, 
#     "B": 256,
#     "Sc": 256,
#     "Slice": 2,
#     "H": 512,
#     "P": 3,
#     "norm": "gLN", #BN, gLN, cLN
#     "num_spks": num_spks,
#     "non_linear": "sigmoid"
# }


nnet_conf = {
    "L": 16,
    "N": 512,
    "X": 8,
    "R": 2, 
    "B": 256,
    "Sc": 256,
    # "Slice": 1,
    "H": 512,
    "P": 3,
    "norm": "gLN", #BN, gLN, cLN
    "num_spks": num_spks,
    "non_linear": "sigmoid"
}

# data configure:
# train_dir = "/media/denoiser/66270bea-3081-4132-b76d-84f0ac1e156e/speech_donoiser_new/datasets/ner-300hr/tr/"
# dev_dir = "/media/denoiser/66270bea-3081-4132-b76d-84f0ac1e156e/speech_donoiser_new/datasets/ner-300hr/cv/"


# 請填寫 tr cv scp 位置
train_dir = "/content/drive/MyDrive/Speech_Denoising/dataset/final_dataset/tr/"
dev_dir = "/content/drive/MyDrive/Speech_Denoising/dataset/final_dataset/cv/"


# 輸入資料用.scp
#.scp 檔案（Script file）：這是 Kaldi 語音工具包中標準的檔案格式。裡面不是真正的音檔，而是一個純文字檔，記錄著 [音檔ID] [實際硬碟路徑]。
# mix_scp：輸入給模型的「混合音訊」（也就是包含噪音或兩人重疊說話的聲音）
# ref_scp：模型的「標準答案」（Ground Truth），即乾淨的說話者 1 (spk1) 與說話者 2 (spk2) 的音訊。
train_data = {
    "mix_scp":
    train_dir + "mix.scp",
    "ref_scp":
    [train_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate":
    fs,
}

dev_data = {
    "mix_scp": dev_dir + "mix.scp",
    "ref_scp":
    [dev_dir + "spk{:d}.scp".format(n) for n in range(1, 1 + num_spks)],
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    # "lr": 1e-3,
    'lr': 0.001,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 200,  # batch number
    "no_impr":100,
    "loss_mode": "sisnr"
}
