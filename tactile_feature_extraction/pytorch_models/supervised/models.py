import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_model_summary import summary
# from vit_pytorch.vit import ViT


def create_model(
    in_dim,
    in_channels,
    out_dim,
    model_params,
    saved_model_dir=None,
    device='cpu',
    cnn_model_dir=None
):

    if model_params['model_type'] in ['simple_cnn', 'posenet_cnn']:
        model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif model_params['model_type'] == 'nature_cnn':
        model = NatureCNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif model_params['model_type'] == 'resnet':
        model = ResNet(
            ResidualBlock,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs'],
        ).to(device)
        '''
        elif model_params['model_type'] == 'vit':
            model = ViT(
                image_size=in_dim[0],
                channels=in_channels,
                num_classes=out_dim,
                **model_params['model_kwargs']
            ).to(device)
        '''
    elif model_params['model_type'] == 'lstm':
        model = LSTMModel(
            input_dim=2048,
            hidden_dim=64,
            output_dim=out_dim,
            num_layers=2
        ).to(device)
    elif model_params['model_type'] == 'transformer':
        model = TransformerModel(
            input_dim=2048,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            output_dim=out_dim
        ).to(device)
    elif model_params['model_type'] == 'conv_lstm':
        model = ConvLstm(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            lock_cnn=False, # 锁定CNN层
            lstm_hidden_dim=128,
            lstm_layers=2,
            **model_params['model_kwargs']
        ).to(device)
    elif model_params['model_type'] == 'conv_transformer':
        model = ConvTransformer(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            d_model=512,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            full_transformer=False,
            **model_params['model_kwargs']
        ).to(device)
    elif model_params['model_type'] == 'conv_gru':
        model = ConvGRU(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            lock_cnn=False, # 锁定CNN层
            gru_hidden_dim=128,
            gru_layers=2,
            cnn_pretained=cnn_model_dir,
            **model_params['model_kwargs']
        ).to(device)
    elif model_params['model_type'] == 'seq2seq_gru':
        model = Seq2SeqGRU(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            lock_cnn=False, # 锁定CNN层
            gru_hidden_dim=128,
            gru_layers=2,
            cnn_pretained=cnn_model_dir,
            **model_params['model_kwargs']
        ).to(device)
    elif model_params['model_type'] == 'CNN3D':
        model = CNN3D(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            n_frames=5,
            **model_params['model_kwargs']
        ).to(device)
    elif model_params['model_type'] == 'seq2seq_gru_attention':
        model = Seq2SeqGRUAttention(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            lock_cnn=False, # 锁定CNN层
            gru_hidden_dim=128,
            gru_layers=2,
            **model_params['model_kwargs']
        ).to(device)

    else:
        raise ValueError('Incorrect model_type specified:  %s' % (model_params['model_type'],))

    if saved_model_dir is not None:
        model.load_state_dict(torch.load(os.path.join(
            saved_model_dir, 'best_model.pth'), map_location='cpu')
        )
    '''
    print(summary(
        model,
        torch.zeros((1, in_channels, *in_dim)).to(device),
        show_input=True
    ))
    '''
    return model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CNN(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        conv_layers=[16, 16, 16],
        conv_kernel_sizes=[5, 5, 5],
        fc_layers=[128, 128],
        activation='relu',
        apply_batchnorm=False,
        dropout=0.0,
    ):
        super(CNN, self).__init__()

        assert len(conv_layers) > 0, "conv_layers must contain values"
        assert len(fc_layers) > 0, "fc_layers must contain values"
        assert len(conv_layers) == len(conv_kernel_sizes), "conv_layers must be same len as conv_kernel_sizes"

        # add first layer to network
        cnn_modules = []
        cnn_modules.append(nn.Conv2d(in_channels, conv_layers[0], kernel_size=conv_kernel_sizes[0], stride=1, padding=2))
        if apply_batchnorm:
            cnn_modules.append(nn.BatchNorm2d(conv_layers[0]))
        cnn_modules.append(nn.ReLU())
        cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # add the remaining conv layers by iterating through params
        for idx in range(len(conv_layers) - 1):
            cnn_modules.append(
                nn.Conv2d(
                    conv_layers[idx],
                    conv_layers[idx + 1],
                    kernel_size=conv_kernel_sizes[idx + 1],
                    stride=1, padding=2)
                )

            if apply_batchnorm:
                cnn_modules.append(nn.BatchNorm2d(conv_layers[idx+1]))

            if activation == 'relu':
                cnn_modules.append(nn.ReLU())
            elif activation == 'elu':
                cnn_modules.append(nn.ELU())
            cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # create cnn component of network
        self.cnn = nn.Sequential(*cnn_modules)

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        fc_modules.append(nn.ReLU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            if activation == 'relu':
                fc_modules.append(nn.ReLU())
            elif activation == 'elu':
                fc_modules.append(nn.ELU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class CNN3D(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        n_frames,
        conv_layers=[16, 16, 16],
        conv_kernel_sizes=[5, 5, 5],
        fc_layers=[128, 128],
        activation='relu',
        apply_batchnorm=False,
        dropout=0.0,
    ):
        super(CNN3D, self).__init__()

        assert len(conv_layers) > 0, "conv_layers must contain values"
        assert len(fc_layers) > 0, "fc_layers must contain values"
        assert len(conv_layers) == len(conv_kernel_sizes), "conv_layers must be same len as conv_kernel_sizes"

        # add first layer to network
        cnn_modules = []
        cnn_modules.append(nn.Conv3d(in_channels, conv_layers[0], kernel_size=conv_kernel_sizes[0], stride=1, padding=2))
        if apply_batchnorm:
            cnn_modules.append(nn.BatchNorm3d(conv_layers[0]))
        cnn_modules.append(nn.ReLU())
        cnn_modules.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        # add the remaining conv layers by iterating through params
        for idx in range(len(conv_layers) - 1):
            cnn_modules.append(
                nn.Conv3d(
                    conv_layers[idx],
                    conv_layers[idx + 1],
                    kernel_size=conv_kernel_sizes[idx + 1],
                    stride=1, padding=2)
                )

            if apply_batchnorm:
                cnn_modules.append(nn.BatchNorm3d(conv_layers[idx+1]))

            if activation == 'relu':
                cnn_modules.append(nn.ReLU())
            elif activation == 'elu':
                cnn_modules.append(nn.ELU())
            cnn_modules.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        # create cnn component of network
        self.cnn = nn.Sequential(*cnn_modules)

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels,n_frames, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        fc_modules.append(nn.ReLU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            if activation == 'relu':
                fc_modules.append(nn.ReLU())
            elif activation == 'elu':
                fc_modules.append(nn.ELU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        # x batch_size, timesteps, channel_x, h_x, w_x
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, channel_x, timesteps, h_x, w_x]
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper (Commonly used in RL):
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    """

    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        fc_layers=[128, 128],
        dropout=0.0
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        fc_modules.append(nn.ReLU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, in_channels, layers, out_dim):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512, out_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只用最后一个时间步的输出进行预测
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # 定义GRU层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 使用GRU进行前向传播
        out, _ = self.gru(x, h0)
        # out: [batch_size, seq_length, hidden_dim]

        # 只使用最后一个时间步的输出进行预测
        out = self.fc(out[:, -1, :])
        # out: [batch_size, output_dim]
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )
        self.fc1 = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc1(output[:, -1, :])
        output = self.activation(output)
        output = self.fc_out(output)
        return output

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, hidden = self.gru(x)
        return hidden

class GRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='None'):
        super(GRUDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        # self.gru = nn.GRU(output_dim+hidden_dim, hidden_dim, batch_first=False)
        self.gru = nn.GRU(output_dim , hidden_dim, batch_first=False)
        # self.fc = nn.Linear(output_dim+2*hidden_dim, output_dim) # x现在是上一步的输出
        # self.fc = nn.Linear(hidden_dim, output_dim)  # x现在是上一步的输出
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
        self.dropout = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    '''
    def forward(self, x, hidden, context):
        # x: [batch_size, 1, output_dim] 前一步的输出
        x=x.permute(1, 0, 2) # [1, batch_size, output_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        x_concat = torch.cat((x, context), dim=2) # [1, batch_size, input_dim+hidden_dim]
        out, hidden = self.gru(x_concat, hidden)

        out = torch.cat((x.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1) # [batch_size, input_dim+2*hidden_dim]
        out = self.fc(out)
        return out, hidden
    '''

    def forward(self, x, hidden):
        # x: [batch_size, 1, output_dim] 前一步的输出
        x = x.permute(1, 0, 2)  # [1, batch_size, output_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[0])
        if self.activation:
            out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out, hidden
class Seq2SeqGRU(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channels,
            out_dim,
            lock_cnn=False,
            gru_hidden_dim=128,
            gru_layers=2,
            conv_layers=[16, 16, 16],
            conv_kernel_sizes=[5, 5, 5],
            fc_layers=[128, 128],
            activation='relu',
            apply_batchnorm=False,
            dropout=0.0,
            cnn_pretained=None
        ):
        super(Seq2SeqGRU, self).__init__()
        self.conv_model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            conv_layers=conv_layers,
            conv_kernel_sizes=conv_kernel_sizes,
            fc_layers=fc_layers,
            activation=activation,
            apply_batchnorm=apply_batchnorm,
            dropout=dropout
        )
        if cnn_pretained:
            print('load cnn model from %s' % cnn_pretained)
            self.conv_model.load_state_dict(torch.load(cnn_pretained))
        else:
            lock_cnn = False
        if lock_cnn:
            for param in self.conv_model.parameters():
                param.requires_grad = False
        # 移除最后一层全连接层
        self.conv_model.fc = nn.Sequential(*list(self.conv_model.fc.children())[:-1])
        self.out_dim = out_dim
        self.encoder = GRUEncoder(input_dim=fc_layers[-1], hidden_dim=gru_hidden_dim)
        self.decoder = GRUDecoder(input_dim=fc_layers[-1], hidden_dim=gru_hidden_dim, output_dim=out_dim, activation='relu')
    def forward(self, x, output_last=True,target=None):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        outputs = torch.zeros(batch_size, timesteps, self.out_dim).to(x.device) # [batch_size, timesteps, out_dim]
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        gru_input = conv_output.view(batch_size, timesteps, -1)
        hidden = self.encoder(gru_input)
        context=hidden
        # x 全零向量
        x = torch.zeros(batch_size, 1, self.out_dim).to(x.device)
        for t in range(0, timesteps):
            # output, hidden = self.decoder(x, hidden, context)
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            if target is not None:
                x = target[:, t, :].unsqueeze(1)
            else:
                x = output.unsqueeze(1) # [batch_size, 1, out_dim]

        # outputs shape [batch_size, timesteps, out_dim]
        # labels shape [batch_size, out_dim,timesteps]
        # outputs = outputs.permute(0, 2, 1)
        if output_last:
            return output
        else:
            return outputs


class FullTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_dim, dropout=0.1):
        super(FullTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Embedding layers
        self.src_embedding = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Linear(output_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder and Decoder Layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_decoder_layers
        )
        # Output linear layer
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # Source and target embeddings
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Apply positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        # Pass through the encoder and decoder
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)

        # Apply the output layer
        output = self.fc_out(output)
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConvLstm(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channels,
            out_dim,
            lock_cnn=False,
            lstm_hidden_dim=128,
            lstm_layers=2,
            conv_layers=[16, 16, 16],
            conv_kernel_sizes=[5, 5, 5],
            fc_layers=[128, 128],
            activation='relu',
            apply_batchnorm=False,
            dropout=0.0,
            cnn_pretained=None,
        ):
        super(ConvLstm, self).__init__()
        self.conv_model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            conv_layers=conv_layers,
            conv_kernel_sizes=conv_kernel_sizes,
            fc_layers=fc_layers,
            activation=activation,
            apply_batchnorm=apply_batchnorm,
            dropout=dropout
        )
        if cnn_pretained:
            print('load cnn model from %s' % cnn_pretained)
            self.conv_model.load_state_dict(torch.load(cnn_pretained))
        else:
            lock_cnn = False
        if lock_cnn:
            for param in self.conv_model.parameters():
                param.requires_grad = False
        # 移除最后一层全连接层
        self.conv_model.fc = nn.Sequential(*list(self.conv_model.fc.children())[:-1])
        self.Lstm = LSTMModel(
            input_dim=fc_layers[-1],
            hidden_dim=lstm_hidden_dim,
            output_dim=out_dim,
            num_layers=lstm_layers
        )
        # self.output_layer = nn.Linear(lstm_hidden_dim, out_dim)
        self.fc1 = nn.Linear(lstm_hidden_dim, 64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(64, out_dim)
    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        output = self.Lstm(lstm_input) # [batch_size, timesteps, lstm_hidden_dim]
        # lstm_output = lstm_output[:, -1, :]
        # output = self.output_layer(lstm_output)
        #output = self.fc1(output)
        #output = self.activation(output)
        # output = self.fc2(output)
        return output

class ConvGRU(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channels,
            out_dim,
            lock_cnn=False,
            gru_hidden_dim=128,
            gru_layers=2,
            conv_layers=[16, 16, 16],
            conv_kernel_sizes=[5, 5, 5],
            fc_layers=[128, 128],
            activation='relu',
            apply_batchnorm=False,
            dropout=0.0,
            cnn_pretained=None,
        ):
        super(ConvGRU, self).__init__()
        self.conv_model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            conv_layers=conv_layers,
            conv_kernel_sizes=conv_kernel_sizes,
            fc_layers=fc_layers,
            activation=activation,
            apply_batchnorm=apply_batchnorm,
            dropout=dropout
        )
        if cnn_pretained:
            print('load cnn model from %s' % cnn_pretained)
            self.conv_model.load_state_dict(torch.load(cnn_pretained))
        else:
            lock_cnn = False
        if lock_cnn:
            for param in self.conv_model.parameters():
                param.requires_grad = False
        # 移除最后一层全连接层
        self.conv_model.fc = nn.Sequential(*list(self.conv_model.fc.children())[:-1])
        self.GRU = GRUModel(
            input_dim=fc_layers[-1],
            hidden_dim=gru_hidden_dim,
            output_dim=gru_hidden_dim,
            num_layers=gru_layers
        )
        self.fc = nn.Linear(gru_hidden_dim, 64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(64, out_dim)
        # self.output_layer = nn.Linear(lstm_hidden_dim, out_dim)
    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        gru_input = conv_output.view(batch_size, timesteps, -1)
        output = self.GRU(gru_input)
        print(output.shape)
        output = self.fc(output)
        output = self.activation(output)
        output = self.fc2(output)
        return output

class ConvGRUAttention(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channels,
            out_dim,
            lock_cnn=False,
            gru_hidden_dim=128,
            gru_layers=2,
            conv_layers=[16, 16, 16],
            conv_kernel_sizes=[5, 5, 5],
            fc_layers=[128, 128],
            activation='relu',
            apply_batchnorm=False,
            dropout=0.0,
            cnn_pretained=None,
        ):
        super(ConvGRUAttention, self).__init__()
        self.conv_model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            conv_layers=conv_layers,
            conv_kernel_sizes=conv_kernel_sizes,
            fc_layers=fc_layers,
            activation=activation,
            apply_batchnorm=apply_batchnorm,
            dropout=dropout
        )
        if cnn_pretained:
            print('load cnn model from %s' % cnn_pretained)
            self.conv_model.load_state_dict(torch.load(cnn_pretained))
        else:
            lock_cnn = False
        if lock_cnn:
            for param in self.conv_model.parameters():
                param.requires_grad = False
        # 移除最后一层全连接层
        self.conv_model.fc = nn.Sequential(*list(self.conv_model.fc.children())[:-1])
        self.GRU = GRUModel(
            input_dim=fc_layers[-1],
            hidden_dim=gru_hidden_dim,
            output_dim=gru_hidden_dim,
            num_layers=gru_layers
        )

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc2 = nn.Linear(gru_hidden_dim, out_dim)
        # self.output_layer = nn.Linear(lstm_hidden_dim, out_dim)
    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        gru_input = conv_output.view(batch_size, timesteps, -1)
        output = self.GRU(gru_input)

        return output

'''
class Seq2SeqConvGRU(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channels,
            out_dim,
            lock_cnn=False,
            gru_hidden_dim=128,
            gru_layers=2,
            conv_layers=[16, 16, 16],
            conv_kernel_sizes=[5, 5, 5],
            fc_layers=[128,128],
            activation='relu',
            apply_batchnorm=False,
            dropout=0.0,
            cnn_pretained=None,
    ):
        super(Seq2SeqConvGRU, self).__init__()
        self.conv_model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            conv_layers=conv_layers,
            conv_kernel_sizes=conv_kernel_sizes,
            fc_layers=fc_layers,
            activation=activation,
            apply_batchnorm=apply_batchnorm,
            dropout=dropout
        )
        if cnn_pretained:
            self.conv_model.load_state_dict(torch.load(cnn_pretained))
        if lock_cnn:
            for param in self.conv_model.parameters():
                param.requires_grad = False
        self.conv_model.fc = nn.Sequential(*list(self.conv_model.fc.children())[:-1])

        self.hidden_dim = gru_hidden_dim
        self.num_layers = gru_layers

        self.encoder = nn.GRU(
            input_size=fc_layers[-1],
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )
        self.decoder = nn.GRU(
            input_size=gru_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(gru_hidden_dim, out_dim)

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        gru_input = conv_output.view(batch_size, timesteps, -1)
        encoder_output,h0 = self.encoder(gru_input, h0)
        # Decoding step (you might want to use encoder_output in a specific way here)
        output,_ = self.decoder(encoder_output,h0)
        output = self.fc_out(output[:, -1, :])  # Take only the output of the last timestep

        return output
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
'''

class ConvTransformer(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channels,
            out_dim,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            full_transformer=False,
            conv_layers=[16, 16, 16],
            conv_kernel_sizes=[5, 5, 5],
            fc_layers=[128, 128],
            activation='relu',
            apply_batchnorm=False,
            dropout=0.1,
            cnn_pretained=None
        ):
        super(ConvTransformer, self).__init__()
        self.conv_model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            conv_layers=conv_layers,
            conv_kernel_sizes=conv_kernel_sizes,
            fc_layers=fc_layers,
            activation=activation,
            apply_batchnorm=apply_batchnorm,
            dropout=dropout
        )
        if cnn_pretained:
            self.conv_model.load_state_dict(torch.load(cnn_pretained))
        # 移除最后一层全连接层
        self.conv_model.fc = nn.Sequential(*list(self.conv_model.fc.children())[:-1])
        if full_transformer:
            self.transformer = FullTransformerModel(
                input_dim=fc_layers[-1],
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                output_dim=out_dim
            )
        else:
            self.transformer = TransformerModel(
                input_dim=fc_layers[-1],
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                output_dim=out_dim
            )

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        transformer_input = conv_output.view(batch_size, timesteps, -1)
        output = self.transformer(transformer_input)
        return output


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        '''
        self.attn_fc = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim
        )
        '''
        self.attn_fc = nn.Linear(
            encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim
        )
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [src length, batch size, encoder hidden dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        assert src_length == 5
        assert hidden.shape == torch.Size([batch_size, 128])
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src length, decoder hidden dim]
        assert hidden.shape == torch.Size([batch_size, 5, 128])
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src length, decoder hidden dim]
        attention = self.v_fc(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1)

class GRUEncoderAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUEncoderAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True)
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=False)
        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [src length, batch size, input_dim]
        out, hidden = self.gru(x)
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))) # 目的是
        # hidden: [num_layers, batch_size, hidden_dim]
        return out,hidden

class GRUDecoderAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='None'):
        super(GRUDecoderAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = Attention(hidden_dim, hidden_dim)
        self.gru = nn.GRU(output_dim + hidden_dim , hidden_dim, batch_first=False)
        self.fc = nn.Linear(2* hidden_dim + output_dim, hidden_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, endoder_outputs):
        # x: [batch_size, 1, output_dim] 前一步的输出
        x = x.permute(1, 0, 2)  # [1, batch_size, output_dim]
        a = self.attention(hidden, endoder_outputs)
        a = a.unsqueeze(1) # [batch_size, 1, src length]
        # endoder_outputs: [src length, batch size, encoder hidden dim * 2]
        endoder_outputs = endoder_outputs.permute(1, 0, 2)  # [batch size, src length, encoder hidden dim * 2]
        weighted = torch.bmm(a, endoder_outputs) # [batch size, 1, encoder hidden dim * 2]
        weighted = weighted.permute(1, 0, 2) # [1, batch size, encoder hidden dim * 2]
        x_concat = torch.cat((x, weighted), dim=2) # [1, batch size, output_dim + encoder hidden dim * 2]
        # hidden: [num_layers, batch_size, hidden_dim]
        out, hidden = self.gru(x_concat, hidden.unsqueeze(0))
        assert (out == hidden).all()
        x = x.squeeze(0)
        out = out.squeeze(0)
        weighted = weighted.squeeze(0)
        out = torch.cat((out, weighted,x), dim=1)
        # out = [batch_size, hidden_dim * 3 + output_dim]
        out = self.fc(out)
        if self.activation:
            out = self.activation(out)

        out = self.fc2(out)
        # hidden = [num_layers, batch_size, hidden_dim]
        return out, hidden.squeeze(0), a.squeeze(1)

class Seq2SeqGRUAttention(nn.Module):
    def __init__(self,
                 in_dim,
                 in_channels,
                 out_dim,
                 lock_cnn=False,
                 gru_hidden_dim=128,
                 gru_layers=2,
                 conv_layers=[16, 16, 16],
                 conv_kernel_sizes=[5, 5, 5],
                 fc_layers=[128, 128],
                 activation='relu',
                 apply_batchnorm=False,
                 dropout=0.0):
        super(Seq2SeqGRUAttention, self).__init__()
        self.conv_model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            conv_layers=conv_layers,
            conv_kernel_sizes=conv_kernel_sizes,
            fc_layers=fc_layers,
            activation=activation,
            apply_batchnorm=apply_batchnorm,
            dropout=dropout
        )
        if lock_cnn:
            for param in self.conv_model.parameters():
                param.requires_grad = False
        # 移除最后一层全连接层
        self.conv_model.fc = nn.Sequential(*list(self.conv_model.fc.children())[:-1])
        self.out_dim = out_dim
        self.encoder = GRUEncoderAttention(input_dim=fc_layers[-1], hidden_dim=gru_hidden_dim)
        self.decoder = GRUDecoderAttention(input_dim=fc_layers[-1], hidden_dim=gru_hidden_dim, output_dim=out_dim, activation='relu')
    def forward(self, x, output_last=True,target=None):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        # 检验输入数据的维度是否正确
        assert timesteps == 5
        assert channel_x == 1
        outputs = torch.zeros(batch_size, timesteps, self.out_dim).to(x.device)
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        gru_input = conv_output.view(timesteps, batch_size,  -1) # [src length, batch size, input_dim]
        encoder_outputs, hidden = self.encoder(gru_input)
        hidden = hidden.squeeze(0)
        # x 全零向量
        x = torch.zeros(batch_size, 1, self.out_dim).to(x.device)
        for t in range(0, timesteps):
            output, hidden, _ = self.decoder(x, hidden, encoder_outputs)


            outputs[:, t, :] = output
            if target is not None:
                x = target[:, t, :].unsqueeze(1)
            else:
                x = output.unsqueeze(1)
        if output_last:
            return output
        else:
            return outputs


