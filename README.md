# SadTalker-Video-Lip-Sync

<a target="_blank" href="<a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/drive/1PqPhV_MgiPKr729DG9w1B0DW5zaxIkGd?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

本プロジェクトは、SadTalkersに基づいてWav2lipを実現する動画唇形合成のためのものです。音声駆動で唇形を動画ファイルとして生成し、顔の領域を強化して合成する方法を設定することで、生成された唇形の鮮明度を向上させます。生成された動画に対して、DAIN補間アルゴリズムを使用してフレームを補完し、合成唇形の動きの過渡を補強することで、唇形をよりスムーズでリアルかつ自然に見せます。

## 1.環境準備 (Environment)

```python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install ffmpeg
pip install -r requirements.txt

# DAINモデルを使用してフレーム補完を行う場合は、paddleをインストール
# CUDA 11.2
python -m pip install paddlepaddle-gpu==2.3.2.post112 \
-f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## 2.プロジェクト構造 (Repository structure)

```
SadTalker-Video-Lip-Sync
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├── ...
├──dian_output
|   ├── ...
├──examples
|   ├── audio
|   ├── video
├──results
|   ├── ...
├──src
|   ├── ...
├──sync_show
├──third_part
|   ├── ...
├──...
├──inference.py
├──README.md
```

## 3.モデル推論 (Inference)

```python
python inference.py --driven_audio <audio.wav> \
                    --source_video <video.mp4> \
                    --enhancer <none,lip,face> \  #(デフォルトはlip)
                    --use_DAIN \ #(この機能を使用すると大量のメモリと時間を消費します)
                    --time_step 0.5 #(補間フレーム頻度、デフォルトは0.5、つまり25fpsから50fpsへ; 0.25の場合、25fpsから100fpsへ)
```



## 4.合成効果 (Results)

```python
# 合成効果は./sync_showディレクトリに表示されます：
# original.mp4 元の動画
# sync_none.mp4 何も強化されていない合成効果
# none_dain_50fps.mp4 DAINモデルのみを使用して25fpsを50fpsに補完
# lip_dain_50fps.mp4 唇形領域を強化して唇形を鮮明にし、DAINモデルを使用して25fpsを50fpsに補完
# face_dain_50fps.mp4 顔全体の領域を強化して唇形を鮮明にし、DAINモデルを使用して25fpsを50fpsに補完

# 以下は異なる方法で生成された動画の比較です
# our.mp4 本プロジェクトSadTalker-Video-Lip-Syncで生成された動画
# sadtalker.mp4 SadTalkerで生成された完全な動画
# retalking.mp4 ReTalkingで生成された動画
# wav2lip.mp4 Wav2Lipで生成された動画
```

https://user-images.githubusercontent.com/52994134/231769817-8196ef1b-c341-41fa-9b6b-63ad0daf14ce.mp4

動画を結合するとフレーム数が25fpsに統一されるため、補間効果の違いは分かりません。具体的な詳細は./sync_showディレクトリ内の個別動画を比較してください。

**本プロジェクトとSadTalker、ReTalking、Wav2Lipの唇形合成の効果比較：**

|                           **our**                            |                        **sadtalker**                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <video  src="https://user-images.githubusercontent.com/52994134/233003969-91fa9e94-a958-4e2d-b958-902cc7711b8a.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003985-86d0f75c-d27f-4a52-ac31-2649ccd39616.mp4" type="video/mp4"> </video> |
|                        **retalking**                         |                         **wav2lip**                          |
| <video  src="https://user-images.githubusercontent.com/52994134/233003982-2fe1b33c-b455-4afc-ab50-f6b40070e2ca.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003990-2f8c4b84-dc74-4dc5-9dad-a8285e728ecb.mp4" type="video/mp4"> </video> |

READMEで表示されている動画はリサイズされています。元の動画は./sync_showディレクトリ内の異なるカテゴリの合成動画を比較してください。

## 5.事前学習済みモデル (Pretrained model)

事前学習済みモデルは以下の通りです：

```python
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├──auido2exp_00300-model.pth
|   ├──auido2pose_00140-model.pth
|   ├──epoch_20.pth
|   ├──facevid2vid_00189-model.pth.tar
|   ├──GFPGANv1.3.pth
|   ├──GPEN-BFR-512.pth
|   ├──mapping_00109-model.pth.tar
|   ├──ParseNet-latest.pth
|   ├──RetinaFace-R50.pth
|   ├──shape_predictor_68_face_landmarks.dat
|   ├──wav2lip.pth
```

事前学習済みモデルのチェックポイントダウンロードパス：

Google Drive：https://drive.google.com/file/d/1iS4LzBOxXUZs0r9hEGs9OIFgbH6ygAUh/view?usp=sharing

```python
# ダウンロードした圧縮ファイルをプロジェクトのパスに解凍します（GoogleドライブおよびQuark网盘からダウンロードした場合は実行が必要です）
cd SadTalker-Video-Lip-Sync
tar -zxvf checkpoints.tar.gz
```

## 参考(Reference)

- SadTalker:https://github.com/Winfredy/SadTalker
-  VideoReTalking：https://github.com/vinthony/video-retalking
- DAIN :https://arxiv.org/abs/1904.00830
- PaddleGAN:https://github.com/PaddlePaddle/PaddleGAN
