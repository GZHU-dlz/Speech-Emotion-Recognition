from transformers import Wav2Vec2FeatureExtractor, WavLMModel
# from funasr import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from emodb_spec import SER_AlexNet,TemporalMfccAwareBlock
from emodb_spec import ECAPA_TDNN
from transformerss.transformer_timm import AttentionBlock
from emodb_spec import Tiss
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
# from models.ser_spec import Simam_module
from emodb_spec import SE_Block
from emodb_spec import BiLSTM_SA
from sklearn.decomposition import PCA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 此模块内是关于spec与mfcc与wav2vec2的处理，并最终归并后的最后结果。
# __all__ = ['Ser_Model']
class AlphaGenerator(nn.Module):
    def __init__(self, d):
        super(AlphaGenerator, self).__init__()
        self.fc1 = nn.Linear(d,d)
        self.bn = nn.BatchNorm1d(d)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, d]
        x = self.fc1(x)
        # if x.shape[0]==64:
        #     x=self.bn(x)
        # elif x.shape[0]==45:
        #     x=self.bn2(x)
        # else:
        #     x=self.bn3(x)
        x=self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        alpha = self.sigmoid(x)  # Output: [B, 1], scalar between 0~1
        return alpha
class Adapters(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Adapters, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.layer(x)

class FeatureAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.fc=nn.Linear(input_dim, output_dim)
    def forward(self, x):
        # x: [batch, seq_len, feat_dim]
        # 1. 时间维度注意力池化
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        context_vector = torch.sum(attn_weights * x, dim=1)  # [batch, feat_dim]

        # 2. 降维
        output = self.fc(context_vector)  # [batch, output_dim]
        return   output.unsqueeze(0)# 添加批次外维度 [1, batch, output_dim] [64,1024]
class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class LSKblock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv1d(dim, dim//2, 1)
        self.conv2 = nn.Conv1d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv1d(2, 2, kernel_size=7, padding=3)
        self.conv = nn.Conv1d(dim//2, dim, 1)

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)

        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0:1,:] + attn2 * sig[:,1:2,:]
        attn = self.conv(attn)

        # 回到 [B, T, F]
        return (x * attn).permute(0, 2, 1)
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # self.conv_spatial_extra = nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=16, groups=dim, dilation=4)

        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        # attn3 = self.conv_spatial_extra(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        # attn3 = self.conv2(attn3)

        #两个深度可分离卷积
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        # temp=(attn2 * sig[: ,1 ,: ,:].unsqueeze(1)+attn3 * sig[: ,1 ,: ,:].unsqueeze(1))/2
        attn = attn1 * sig[: ,0 ,: ,:].unsqueeze(1) + attn2 * sig[: ,1 ,: ,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim//reduction, 1)
        self.fc2 = nn.Conv2d(dim//reduction, dim, 1)
    def forward(self, x):
        # 全局时–频池化
        w = x.mean(dim=[2,3], keepdim=True)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w
class EmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.stem = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        # )
        # self.LSK = LSKblock(64)
        # self.SEblock = SEBlock(64)
        # self.ECAblock=eca_layer()
        self.LSK = LSKblock1D(1024)
        # 降到时序轴
        self.pool_freq = nn.AdaptiveAvgPool2d((1, None))  # 只池化频率轴
        self.temporal = nn.Conv1d(64, 64, 3, padding=1, groups=64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(64, 32), nn.ReLU(inplace=True),
        #     nn.Dropout(0.3), nn.Linear(32, num_classes)
        # )
        self.conv1 = nn.Conv1d(1024, 64, kernel_size=3, padding=1)
        self.pool1 = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        # x = self.stem(x)                  # [B,64,256,384]
        x = self.LSK(x)                  # [B,64,256,384]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.pool1(x).squeeze(-1)
        # x=self.SEblock(x)
        # x = self.pool_freq(x).squeeze(2)  # [B,64,384]
        # x = self.temporal(x)              # [B,64,384]
        # x = self.global_pool(x).squeeze(2)# [B,64]
        # return self.classifier(x)
        return x
class Ser_Model(nn.Module):
    def __init__(self):
        super(Ser_Model, self).__init__()
        self.modelss = AttentionBlock(in_dim_k=64, in_dim_q=64, out_dim=64, num_heads=1)
        # avgpool
        # self.avg=nn.AvgPool2d(1)
        # CNN for Spectrogram
        # 使用SR-AlexNet是因为这个模型所用的AlexNet与nn自带AlexNet上，有些参数不同，需要重新定义使用
        # self.alexnet_model = SER_AlexNet(num_classes=4, in_ch=3, pretrained=True)
        # self.post_spec_dropout = nn.Dropout(p=0.1)
        # self.post_spec_layer = nn.Linear(9216, 128)  # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        # LSTM for MFCC(原input_size=40)
        # self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,
        #                          bidirectional=True)  # bidirectional = True

        # self.post_mfcc_dropout = nn.Dropout(p=0.1)
        # self.post_mfcc_layer = nn.Linear(153600,
        #                                  128)  # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        # Spectrogram + MFCC
        # self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        # self.post_spec_mfcc_att_layer = nn.Linear(256, 149)  # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        # self.post_wav_dropout = nn.Dropout(p=0.1)
        # self.post_mfmel_layer=nn.Linear(192,128)
        # self.post_wav_layer = nn.Linear(768, 128)  # 512 for 1 and 768 for 2
        self.post_wav_layer = nn.Linear(768, 128)
        self.post_att_dropout = nn.Dropout(p=0.2)
        self.post_att_layer_1 = nn.Linear(64, 32)
        self.post_att_layer_2 = nn.Linear(32, 16)
        self.post_att_layer_3 = nn.Linear(16, 7)

        # self.melt_layer=nn.Linear(128,64)
        # self.mel_cov1d=nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, dilation=1, padding=0, stride=2)
        # 原来in_channels=300, out_channels=300
        # self.mfcc_conv1d=nn.Conv1d(in_channels=300, out_channels=300, kernel_size=1, dilation=1, padding=0, stride=2)
        # self.TemAware=TemporalAwareBlock(64,2,i,0)
        self.mix = Tiss()

        # SE模块处理
        # self.SEblock=SE_Block(64)
        # self.covn1d=nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        # self.covn2d=nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        # self.linger=nn.Linear(768,256)
        # self.linger1=nn.Linear(256,64)
        self.featuress = FeatureAggregator(input_dim=1024, output_dim=64)
        # self.eca_se=eca_layer()
        self.emotion_spec = EmotionNet()
        # self.convd_audio=nn.Conv1d(in_channels=149, out_channels=149, kernel_size=1,dilation=1,padding=0,stride=2)
        # self.post_audio_layer = nn.Linear(192, 64)
        self.MLP_CID = Adapters(64, 32)
        self.percentage = AlphaGenerator(64)

    def forward(self, audio_spec, audio_mfcc, audio_wav):
        #spec:[64,3,256,384]  audio_wav:[64,48000]
        # outputs_spec = self.emotion_spec(audio_spec)  # 大小为[64,64,384]
        #
        # audio_mel_p = outputs_spec.unsqueeze(0).repeat(1, 1, 1)


        # 利用wavlm-large提取声学特征
        modelss = WavLMModel.from_pretrained(
            "./Wavlm-large",
            torch_dtype=torch.float32,
            weights_only=False
        ).to(device).eval()
        feat_ext = Wav2Vec2FeatureExtractor.from_pretrained("./Wavlm-large")
        # 模拟语音输入
        # 特征提取（关键修改）
        inputs = feat_ext(
            audio_wav,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        # 调整输入形状：从 [1, 1, 64, 48000] -> [64, 48000]
        input_values = inputs.input_values.squeeze(0).squeeze(0)
        # 移动到设备
        input_values = input_values.to(device)
        # 模型前向计算
        with torch.no_grad():
            outputs = modelss(input_values)  # 直接传入处理后的张量
        # 获取特征
        features = outputs.last_hidden_state
        audio_wav=self.emotion_spec(features)

        # 选择注意力机制!!!!!!!!
        # audio_wav = audio_wav.unsqueeze(0)
        # audio_wav = self.modelss(audio_wav, audio_wav)
        # audio_wav = audio_wav.squeeze(0)  # [64,64]



        # audio_wav = self.eca_se(features)
        # audio_wav = self.featuress(features)  # [64,64]

        # 最终特征融合部分(经过transformer)
        # all_feature = self.modelss(audio_mel_p, audio_wav)
        # all_feature = all_feature.squeeze(0)
        # audio_wav = self.modelss(audio_wav, audio_wav)
        # audio_wav = audio_wav.squeeze(0)  # [64,64]
        # # audio_mel_p=audio_mel_p.squeeze(0) #临时加的!!!!!!!!!!
        # # ICD部分(利用audio_mel_p,audio_wav,all_feature都是[1,64,64]大小)
        # audio_wav1 = self.MLP_CID(audio_wav)
        # all_feature1 = self.MLP_CID(all_feature)
        # first_add = all_feature1 + audio_wav1
        # first_melt = self.MLP_CID(first_add)
        # two_add = first_melt + all_feature
        # aa = self.percentage(audio_wav)
        # # all_feature = aa * audio_wav + (1 - aa) * two_add  # [1,64,64]
        # all_feature=aa*audio_wav+(1-aa)*two_add





        # 原分类器方法！！！！！！！！！！！
        # audio_att_d = all_feature.squeeze()#[64,64]
        audio_att_d_1 = self.post_att_dropout(audio_wav)
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)
        audio_att_d_2 = self.post_att_dropout(audio_att_1)  # [batch, 128]
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)
        output_att = self.post_att_layer_3(audio_att_2)  # [batch, 4]

        # F1对应是经过Wav2vec2模型向量与MFCC和Spectrogram结合向量相乘后的结果
        # F2对应三者连接起来后，再经过dropout和relu的结果
        # F3对应F2结果再经过dropout和relu的结果
        # F4、F5对应最终结果
        # 这样做的原因是方便后续将之前的特征拿来进一步融合，所以model返回是多种特征组合
        output = {
            # 'F1': audio_wav_p,
            # 'F2': audio_att_1,
            # 'F3': audio_att_2,
            # 'F4': output_att,
            'M': output_att,
        }

        return output