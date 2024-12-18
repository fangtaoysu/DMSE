import torch
import torch.nn as nn


class MultiModalFusion(nn.Module):
    def __init__(self, text_dim, image_dim, fusion_dim):
        """
        多模态融合模块
        :param text_dim: 文本特征维度
        :param image_dim: 图像特征维度
        :param fusion_dim: 融合特征维度
        """
        super(MultiModalFusion, self).__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)  # 文本特征映射到融合维度
        self.image_proj = nn.Linear(image_dim, fusion_dim)  # 图像特征映射到融合维度
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(fusion_dim, fusion_dim)  # 输出层

    def forward(self, text_features, image_features):
        """
        多模态融合
        :param text_features: 文本特征 [batch_size, text_seq_len, text_dim]
        :param image_features: 图像特征 [batch_size, image_seq_len, image_dim]
        :return: 融合后的特征 [batch_size, seq_len, fusion_dim]
        """
        # 映射到融合维度
        text_features = self.text_proj(text_features)  # [batch_size, text_seq_len, fusion_dim]
        image_features = self.image_proj(image_features)  # [batch_size, image_seq_len, fusion_dim]

        # 拼接特征
        combined_features = torch.cat([text_features, image_features], dim=1)  # [batch_size, seq_len, fusion_dim]

        # 自注意力机制
        fused_features, _ = self.attention(combined_features, combined_features, combined_features)
        return self.fc(fused_features)  # [batch_size, seq_len, fusion_dim]

class MultiGranularityFusion(nn.Module):
    def __init__(self, text_dim, image_dim, fusion_dim):
        """
        粒度级别多模态融合
        :param text_dim: 文本特征维度
        :param image_dim: 图像特征维度
        :param fusion_dim: 融合特征维度
        """
        super(MultiGranularityFusion, self).__init__()
        # 粗粒度融合
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局池化
        self.coarse_fusion = MultiModalFusion(text_dim, image_dim, fusion_dim)

        # 中粒度融合
        self.middle_fusion = MultiModalFusion(text_dim, image_dim, fusion_dim)

        # 细粒度融合
        self.fine_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True)
        self.fine_fc = nn.Linear(fusion_dim, fusion_dim)

        # 最终情境增强
        self.context_transformer = nn.Transformer(d_model=fusion_dim, num_encoder_layers=4)

    def forward(self, text_features, image_features):
        """
        粒度融合过程
        :param text_features: 文本特征 [batch_size, text_seq_len, text_dim]
        :param image_features: 图像特征 [batch_size, image_seq_len, image_dim]
        :return: 最终融合特征 [batch_size, seq_len, fusion_dim]
        """
        # 粗粒度融合
        text_global = self.global_pool(text_features.transpose(1, 2)).squeeze(-1)  # [batch_size, text_dim]
        image_global = self.global_pool(image_features.transpose(1, 2)).squeeze(-1)  # [batch_size, image_dim]
        coarse_output = self.coarse_fusion(text_global.unsqueeze(1), image_global.unsqueeze(1))  # [batch_size, 1, fusion_dim]

        # 中粒度融合
        middle_output = self.middle_fusion(text_features, image_features)  # [batch_size, seq_len, fusion_dim]

        # 细粒度融合
        fine_output, _ = self.fine_attention(text_features, image_features, image_features)  # [batch_size, text_seq_len, fusion_dim]
        fine_output = self.fine_fc(fine_output)

        # 融合所有粒度特征
        combined = torch.cat([coarse_output, middle_output, fine_output], dim=1)  # [batch_size, total_seq_len, fusion_dim]

        # 情境增强
        enhanced_features = self.context_transformer(combined.transpose(0, 1), combined.transpose(0, 1))
        return enhanced_features.transpose(0, 1)  # 恢复到原始维度
